#imports
from .warpset import WarpSet
from .alignmentset import AlignmentSet, Rectangle
from .overlap import Overlap
from .tableio import readtable,writetable
import numpy as np, scipy, matplotlib.pyplot as plt
import os, logging, copy, shutil

#global variables
OVERLAP_FILE_EXT   = '_overlap.csv'
RECTANGLE_FILE_EXT = '_rect.csv'
IM3_EXT='.im3'
IMM_EXT='.imm'
RAW_EXT='.raw'
WARP_EXT='.camWarp_layer'
IMM_FILE_X_SIZE='sizeX'
IMM_FILE_Y_SIZE='sizeY'
IMM_FILE_Z_SIZE='sizeC'
MICROSCOPE_OBJECTIVE_FOCAL_LENGTH=40000. # 20mm in pixels

#set up the logger
logger = logging.getLogger("warpfitter")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s    [%(funcName)s, %(asctime)s]"))
logger.addHandler(handler)

class FittingError(Exception) :
    """
    Class for errors encountered during fitting
    """
    pass

class WarpFitter :
    """
    Main class for fitting a camera matrix and distortion parameters to a set of images based on the results of their alignment
    """
    def __init__(self,samplename,rawfile_dir,metafile_dir,working_dir,overlaps=-1,layer=1,meanimage=None,warpset=None,warp=None) :
        """
        samplename   = name of the microscope data sample to fit to ("M21_1" or equivalent)
        rawfile_dir  = path to directory containing multilayered ".raw" files
        metafile_dir = path to directory containing "dbload" metadata files (assuming at least a "rect.csv" and "overlap.csv")
        working_dir  = path to some local directory to store files produced by the WarpFitter
        overlaps     = list of (or two-element tuple of first/last) #s (n) of overlaps to use for evaluating quality of alignment 
                       (default=-1 will use all overlaps)
        layer        = image layer number (indexed starting at 1) to consider in the warping/alignment (default=1)
        meanimage    = the "meanimage" fit result calculated for the entire dataset (None will recalculate this based on the data subset)
        warpset      = WarpSet object to initialize with (optional, a new WarpSet will be created if None) 
        warp         = CameraWarp object whose optimal parameters will be determined (optional, if None a new one will be created)
        """
        #store the directory paths
        self.samp_name = samplename
        self.rawfile_dir=rawfile_dir
        self.metafile_dir=metafile_dir
        self.working_dir=working_dir
        self.init_dir = os.getcwd()
        self.mean_image = meanimage
        #setup the working directory and the lists of overlaps and rectangles
        self.overlaps, self.rectangles = self.__setupWorkingDirectory(overlaps)
        #get the list of raw file paths
        self.rawfile_paths = [os.path.join(self.rawfile_dir,fn.replace(IM3_EXT,RAW_EXT)) for fn in [r.file for r in self.rectangles]]
        #get the size of the images in the sample
        self.n, self.m, self.nlayers = self.__getImageSizesFromImmFile()
        if layer<1 or layer>self.nlayers :
            raise FittingError(f'Choice of layer ({layer}) is not valid for images with {self.nlayers} layers!')
        #make the warpset object to use
        if warpset is not None :
            self.warpset = warpset
        elif warp is not None :
            if warp.n!=self.n or warp.m!=self.m :
                msg = f'Warp object passed to WarpFitter is set to run on images of size ({warp.n},{warp.m}),'
                msg+=f' not of size ({self.n},{self.m}) as specified by .imm files'
                raise FittingError(msg)
            self.warpset = WarpSet(warp=warp,rawfiles=self.rawfile_paths,nlayers=self.nlayers,layer=layer)
        else :
            self.warpset = WarpSet(n=self.n,m=self.m,rawfiles=self.rawfile_paths,nlayers=self.nlayers,layer=layer)
        #make the alignmentset object to use
        self.alignset = self.__initializeAlignmentSet()
        #the private variable that will hold the best-fit warp
        self.__best_fit_warp = None

    #remove the placeholder files when the object is being deleted
    def __del__(self) :
        logger.info('Removing copied raw layer files....')
        os.chdir(self.working_dir)
        if not os.path.isdir(self.samp_name) :
            os.chdir(self.init_dir)
            return
        try :
            shutil.rmtree(self.samp_name)
        except Exception :
            raise FittingError('Something went wrong in trying to remove the directory of initially-warped files!')
        finally :
            os.chdir(self.init_dir)

    #################### PUBLIC FUNCTIONS ####################

    def loadRawFiles(self) :
        """
        Load the raw files into the warpset, warp/save them, and load them into the alignment set 
        """
        self.warpset.loadRawImageSet(self.rawfile_paths,self.overlaps,self.rectangles)
        self.warpset.warpLoadedImageSet()
        os.chdir(self.working_dir)
        if not os.path.isdir(self.samp_name) :
            os.mkdir(self.samp_name)
        os.chdir(self.samp_name)
        try :
            self.warpset.writeOutWarpedImageSet()
        except Exception :
            raise FittingError('Something went wrong in trying to write out the initial warped files!')
        finally :
            os.chdir(self.init_dir)
        self.alignset.getDAPI(filetype='camWarpDAPI',writeimstat=False,meanimage=self.mean_image)

    def doFit(self,fix_cxcy=False,fix_fxfy=False,fix_k1k2=False,fix_p1p2=False,max_radial_warp=10.,max_tangential_warp=10.,par_bounds=None,
              polish=True,print_every=1,maxiter=1000,show_plots=False) :
        """
        Fit the cameraWarp model to the loaded dataset
        fix_*       = set True to fix groups of parameters
        max_*_warp  = values to use for max warp amount constraints (set to -1 to remove constraints)
        par_bounds  = dictionary of alternate parameter bounds for differential_evolution keyed by name ('cx','cy','fx','fy','k1','k2','p1','p2')
                      can be some parameters or all, will just overwrite the defaults with those supplied
        print_every = print warp parameters and fit results at every [print_every] minimization function calls
        """
        #make the iteration counter and the lists of costs/warp amounts
        self.minfunc_calls=0
        self.costs=[]
        self.max_radial_warps=[]
        self.max_tangential_warps=[]
        #silence the AlignmentSet logger
        logger = logging.getLogger('align')
        logger.setLevel(logging.WARN)
        logger = logging.getLogger('warpfitter')
        #build the list of parameter bounds
        default_bounds = self.__buildDefaultParameterBoundsDict(max_radial_warp,max_tangential_warp)
        parameter_bounds = self.__getParameterBoundsList(par_bounds,default_bounds,fix_cxcy,fix_fxfy,fix_k1k2,fix_p1p2)
        logger.info(f'Floating parameter bounds = {parameter_bounds}')
        #get the list to use to mask fixed parameters in the minimization functions
        self.par_mask = self.__getParameterMask(fix_cxcy,fix_fxfy,fix_k1k2,fix_p1p2)
        #get the list of initial parameters to copy from when necessary
        self.init_pars = self.warpset.getListOfWarpParameters()
        #set the variable describing how often to print progress
        self.print_every = print_every
        #get the array of the initial population
        initial_population=self.__getInitialPopulation(parameter_bounds)
        #get the list of constraints
        constraints = self.__getConstraints(fix_k1k2,fix_p1p2,max_radial_warp,max_tangential_warp)
        #call differential_evolution
        logger.info('Starting initial minimization....')
        self.skip_corners = True
        os.chdir(self.working_dir)
        try :
            firstresult=scipy.optimize.differential_evolution(
                func=self._evalCamWarpOnAlignmentSet,
                bounds=parameter_bounds,
                strategy='best2bin',
                maxiter=maxiter,
                tol=0.025,
                mutation=(0.6,1.00),
                recombination=0.5,
                polish=False,
                init=initial_population,
                constraints=constraints
                )
        except Exception :
            raise FittingError('Something failed in the initial minimization!')
        finally :
            os.chdir(self.init_dir)
        logger.info(f'Initial minimization completed {"successfully" if firstresult.success else "UNSUCCESSFULLY"} in {firstresult.nfev} evaluations.')
        if polish :
            self.skip_corners = False
            #call minimize with trust_constr
            logger.info('Starting polishing minimization....')
            relative_steps = np.array([0.02,0.02,0.02,0.02,0.02,0.02,0.0002,0.0002])[self.par_mask]
            os.chdir(self.working_dir)
            try :
                result=scipy.optimize.minimize(
                    fun=self._evalCamWarpOnAlignmentSet,
                    x0=firstresult.x,
                    method='trust-constr',
                    bounds=parameter_bounds,
                    constraints=constraints,
                    options={'xtol':1e-4,'gtol':1e-3,'finite_diff_rel_step':relative_steps,'maxiter':maxiter}
                    )
            except Exception :
                raise FittingError('Something failed in the polishing minimization!')
            finally :
                os.chdir(self.init_dir)
            msg = f'Final minimization completed {"successfully" if result.success else "UNSUCCESSFULLY"} in {result.nfev} evaluations '
            term_conds = {0:'max iterations',1:'gradient tolerance',2:'parameter tolerance',3:'callback function'}
            msg+=f'due to compliance with {term_conds[result.status]} criteria.'
            logger.info(msg)
        else :
            result=firstresult
        #make the fit progress plots
        self.__makeFitProgressPlots(firstresult.nfev,show_plots)
        #use the fit result to make the best fit warp object and save the figure of its warp fields
        best_fit_pars = self.__correctParameterList(result.x)
        self.warpset.updateCameraParams(best_fit_pars)
        self.__best_fit_warp = copy.deepcopy(self.warpset.warp)
        logger.info(f'Best fit parameters:')
        self.__best_fit_warp.printParams()
        os.chdir(self.working_dir)
        try :
            self.__best_fit_warp.makeWarpAmountFigure()
            self.__best_fit_warp.writeParameterTextFile(self.par_mask)
        except Exception :
            raise FittingError('Something went wrong in trying to save the warping amount figure for the best-fit warp')
        finally :
            os.chdir(self.init_dir)
        #write out the set of alignment comparison images
        self.__makeBestFitAlignmentComparisonImages()
        return result


    #################### FUNCTIONS FOR USE WITH MINIMIZATION ####################

    # !!!!!! For the time being, these functions don't correctly describe dependence on k3, k4, k5, or k6 !!!!!!

    #The function whose return value is minimized by the fitting
    def _evalCamWarpOnAlignmentSet(self,pars) :
        self.minfunc_calls+=1
        #first fix the parameter list so the warp functions always see vectors of the same length
        fixedpars = self.__correctParameterList(pars)
        #update the warp with the new parameters
        self.warpset.updateCameraParams(fixedpars)
        #then warp the images
        self.warpset.warpLoadedImageSet(skip_corners=self.skip_corners)
        #reload the (newly-warped) images into the alignment set
        self.alignset.updateRectangleImages([warpimg for warpimg in self.warpset.images if not (self.skip_corners and warpimg.is_corner_only)])
        #align the images 
        cost = self.alignset.align(skip_corners=self.skip_corners,write_result=False,return_on_invalid_result=True,alreadyalignedstrategy="overwrite")
        #add to the lists to plot
        self.costs.append(cost if cost<1e10 else -0.1)
        self.max_radial_warps.append(self.warpset.warp.maxRadialDistortAmount(fixedpars))
        self.max_tangential_warps.append(self.warpset.warp.maxTangentialDistortAmount(fixedpars))
        #print progress if requested
        if self.minfunc_calls%self.print_every==0 :
            logger.info(self.warpset.warp.paramString())
            msg = f'  Call {self.minfunc_calls} cost={cost}'
            msg+=f' (radial warp={self.max_radial_warps[-1]:.02f}, tangential warp={self.max_tangential_warps[-1]:.02f})'
            logger.info(msg)
        #return the cost from the alignment
        return cost

    #call the warp's max radial distort amount function with the corrected parameters
    def _maxRadialDistortAmountForConstraint(self,pars) :
        return self.warpset.warp.maxRadialDistortAmount(self.__correctParameterList(pars))

    #get the correctly-formatted Jacobian of the maximum radial distortion amount constraint 
    def _maxRadialDistortAmountJacobianForConstraint(self,pars) :
        warpresult = np.array(self.warpset.warp.maxRadialDistortAmountJacobian(self.__correctParameterList(pars)))
        return (warpresult[self.par_mask]).tolist()

    #call the warp's max tangential distort amount function with the corrected parameters
    def _maxTangentialDistortAmountForConstraint(self,pars) :
        return self.warpset.warp.maxTangentialDistortAmount(self.__correctParameterList(pars))

    #get the correctly-formatted Jacobian of the maximum tangential distortion amount constraint 
    def _maxTangentialDistortAmountJacobianForConstraint(self,pars) :
        warpresult = np.array(self.warpset.warp.maxTangentialDistortAmountJacobian(self.__correctParameterList(pars)))
        return (warpresult[self.par_mask]).tolist()

    #################### VISUALIZATION FUNCTIONS ####################

    #function to plot the costs and warps over all the iterations of the fit
    def __makeFitProgressPlots(self,ninitev,show) :
        inititers  = np.linspace(0,ninitev,ninitev,endpoint=False)
        finaliters = np.linspace(ninitev,len(self.costs),len(self.costs)-ninitev,endpoint=False)
        f,ax = plt.subplots(2,3)
        f.set_size_inches(20.,10.)
        ax[0][0].plot(inititers,self.costs[:ninitev])
        ax[0][0].set_xlabel('initial minimization iteration')
        ax[0][0].set_ylabel('cost')
        ax[0][1].plot(inititers,self.max_radial_warps[:ninitev])
        ax[0][1].set_xlabel('initial minimization iteration')
        ax[0][1].set_ylabel('max radial warp')
        ax[0][2].plot(inititers,self.max_tangential_warps[:ninitev])
        ax[0][2].set_xlabel('initial minimization iteration')
        ax[0][2].set_ylabel('max tangential warp')
        ax[1][0].plot(finaliters,self.costs[ninitev:])
        ax[1][0].set_xlabel('final minimization iteration')
        ax[1][0].set_ylabel('cost')
        ax[1][1].plot(finaliters,self.max_radial_warps[ninitev:])
        ax[1][1].set_xlabel('final minimization iteration')
        ax[1][1].set_ylabel('max radial warp')
        ax[1][2].plot(finaliters,self.max_tangential_warps[ninitev:])
        ax[1][2].set_xlabel('final minimization iteration')
        ax[1][2].set_ylabel('max tangential warp')
        os.chdir(self.working_dir)
        try :
            plt.savefig('fit_progress.png')
        except Exception :
            raise FittingError('something went wrong while trying to save the fit progress plots!')
        finally :
            os.chdir(self.init_dir)
        if show :
            plt.show()

    #function to save alignment comparison visualizations in a new directory inside the working directory
    def __makeBestFitAlignmentComparisonImages(self) :
        logger.info('writing out warping/alignment comparison images')
        #make sure the best fit warp exists (which means the warpset is updated with the best fit parameters)
        if self.__best_fit_warp is None :
            raise FittingError('Do not call __makeBestFitAlignmentComparisonImages until after the best fit warp has been set!')
        #start by aligning the raw, unwarped images and getting their shift comparison information/images
        self.alignset.updateRectangleImages(self.warpset.images)
        rawcost = self.alignset.align(write_result=False,alreadyalignedstrategy="overwrite")
        raw_overlap_comparisons_dict = self.alignset.getOverlapComparisonImagesDict()
        #next warp and align the images with the best fit warp
        self.warpset.warpLoadedImageSet()
        self.alignset.updateRectangleImages(self.warpset.images)
        bestcost = self.alignset.align(write_result=False,alreadyalignedstrategy="overwrite")
        warped_overlap_comparisons_dict = self.alignset.getOverlapComparisonImagesDict()
        logger.info(f'Alignment cost from raw images = {rawcost:.08f}; alignment cost from warped images = {bestcost:.08f} ({(100*(1.-bestcost/rawcost)):.04f}% reduction)')
        #write out the overlap comparison figures
        figure_dir_name = 'alignment_overlap_comparisons'
        os.chdir(self.working_dir)
        if not os.path.isdir(figure_dir_name) :
            os.mkdir(figure_dir_name)
        os.chdir(figure_dir_name)
        try :
            for overlap_identifier in raw_overlap_comparisons_dict.keys() :
                code = overlap_identifier[0]
                fn   = overlap_identifier[1]
                pix_to_in = 20./self.n
                if code in [2,8] :
                    f,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
                    f.set_size_inches(self.n*pix_to_in,4.*0.2*self.m*pix_to_in)
                    order = [ax1,ax3,ax2,ax4]
                elif code in [1,3,7,9] :
                    f,ax = plt.subplots(2,2)
                    f.set_size_inches(2.*0.2*self.n*pix_to_in,2*0.2*self.m*pix_to_in)
                    order = [ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
                elif code in [4,6] :
                    f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,sharey=True)
                    f.set_size_inches(4.*0.2*self.n*pix_to_in,self.m*pix_to_in)
                    order=[ax1,ax3,ax2,ax4]
                order[0].imshow(raw_overlap_comparisons_dict[overlap_identifier][0])
                order[0].set_title('raw overlap images')
                order[1].imshow(raw_overlap_comparisons_dict[overlap_identifier][1])
                order[1].set_title('raw overlap images aligned')
                order[2].imshow(warped_overlap_comparisons_dict[overlap_identifier][0])
                order[2].set_title('warped overlap images')
                order[3].imshow(warped_overlap_comparisons_dict[overlap_identifier][1])
                order[3].set_title('warped overlap images aligned')
                plt.savefig(fn)
        except Exception :
            raise FittingError('Something went wrong while trying to write out the overlap comparison images')
        finally :
            os.chdir(self.init_dir)

    #################### PRIVATE HELPER FUNCTIONS ####################

    # helper function to create the working directory and create/write out lists of overlaps and rectangles
    def __setupWorkingDirectory(self,overlaps) :
        #read all the overlaps in this sample's metadata and reduce to what will actually be used
        all_overlaps = readtable(os.path.join(self.metafile_dir,self.samp_name+OVERLAP_FILE_EXT),Overlap)
        if overlaps==-1 :
            olaps = all_overlaps 
        elif isinstance(overlaps,tuple) :
            olaps = all_overlaps[overlaps[0]-1:overlaps[1]]
        elif isinstance(overlaps,list) :
            olaps = [all_overlaps[i-1] for i in overlaps]
        else :
            raise FittingError(f'Cannot recognize overlap choice from overlaps={overlaps} (must be a tuple, list, or -1)')
        if len(olaps)==0 :
            raise FittingError(f'Overlap choice {overlaps} does not represent a valid selection for this sample!')
        #read all the rectangles and store the relevant ones
        all_rectangles = readtable(os.path.join(self.metafile_dir,self.samp_name+RECTANGLE_FILE_EXT),Rectangle)
        rects = [r for r in all_rectangles if r.n in ([o.p1 for o in olaps]+[o.p2 for o in olaps])]
        #create the working directory and write to it the new metadata .csv files
        if not os.path.isdir(self.working_dir) :
            os.mkdir(self.working_dir)
        os.chdir(self.working_dir)
        try :
            writetable(self.samp_name+OVERLAP_FILE_EXT,olaps)
            writetable(self.samp_name+RECTANGLE_FILE_EXT,rects)
        except Exception :
            raise FittingError('Something went wrong in trying to write the truncated overlap/rectangle files in the working directory!')
        finally :
            os.chdir(self.init_dir)
        return olaps, rects

    # helper function to return the (x,y) size of the images read from the .imm file 
    def __getImageSizesFromImmFile(self) :
        first_immfile_path = os.path.join(self.rawfile_dir,self.rectangles[0].file.replace(IM3_EXT,IMM_EXT))
        with open(first_immfile_path) as fp :
            lines=fp.readlines()
        n=int([line.rstrip().split()[1] for line in lines if line.rstrip().split()[0]==IMM_FILE_X_SIZE][0])
        m=int([line.rstrip().split()[1] for line in lines if line.rstrip().split()[0]==IMM_FILE_Y_SIZE][0])
        z=int([line.rstrip().split()[1] for line in lines if line.rstrip().split()[0]==IMM_FILE_Z_SIZE][0])
        return n,m,z

    # helper function to create and return a new alignmentSet object that's set up to run on the identified set of images/overlaps
    def __initializeAlignmentSet(self) :
        a = AlignmentSet(os.path.join(*([os.sep]+self.metafile_dir.split(os.sep)[:-2])),self.working_dir,self.samp_name,interactive=True)
        a.rectanglesoverlaps=self.rectangles, self.overlaps
        if self.mean_image is not None :
            a.meanimage = self.mean_image
        return a

    #helper function to make the list of parameter bounds for fitting
    def __getParameterBoundsList(self,par_bounds,default_bounds,fix_cxcy,fix_fxfy,fix_k1k2,fix_p1p2) :
        parorder=['cx','cy','fx','fy','k1','k2','p1','p2']
        #overwrite the default with anything that was supplied
        bounds_dict = default_bounds
        if par_bounds is not None :
            for name in par_bounds.keys() :
                if name in bounds_dict.keys() :
                    bounds_dict[name] = par_bounds[name]
                else :
                    raise FittingError(f'Parameter "{name}"" in supplied dictionary of bounds not recognized!')
        #remove any parameters that will be fixed
        to_remove = []
        if fix_cxcy :
            to_remove+=['cx','cy']
        if fix_fxfy :
            to_remove+=['fx','fy']
        if fix_k1k2 :
            to_remove+=['k1','k2']
        if fix_p1p2 :
            to_remove+=['p1','p2']
        for parkey in to_remove :
            if parkey in bounds_dict.keys() :
                del bounds_dict[parkey]
        #print info about the parameters that will be used
        fixed_par_string=''
        for name in to_remove :
            fixed_par_string+=name+', '
        msg = f'Will fit with {len(bounds_dict.keys())} parameters'
        if to_remove!=[] :
            msg+=f' ({fixed_par_string[:-2]} fixed).'
        else :
            msg+='.'
        logger.info(msg)
        #return the ordered list of parameters
        return [bounds_dict[name] for name in parorder if name in bounds_dict.keys()]

    #helper function to make the default list of parameter constraints
    def __buildDefaultParameterBoundsDict(self,max_radial_warp,max_tangential_warp) :
        bounds = {}
        # cx/cy bounds are +/- 25% of the center point
        bounds['cx']=(0.75*(self.n/2.),1.25*(self.n/2.))
        bounds['cy']=(0.75*(self.m/2.),1.25*(self.m/2.))
        # fx/fy bounds are +/- 2% of the nominal values 
        bounds['fx']=(0.98*MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
        bounds['fy']=(0.98*MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
        # k1/k2 and p1/p2 bounds are twice those that would produce the max radial and tangential warp, respectively, with all others zero
        # k1 is not allowed to be negative to prevent overfitting, and k2 therefore only has to be its max value on the positive side, not twice.
        maxk1 = self.__findDefaultParameterLimit(4,1,max_radial_warp,self.warpset.warp.maxRadialDistortAmount)
        bounds['k1']=(0.,2.*maxk1)
        maxk2 = self.__findDefaultParameterLimit(5,1000,max_radial_warp,self.warpset.warp.maxRadialDistortAmount)
        bounds['k2']=(-2.*maxk2,maxk2)
        maxp1 = self.__findDefaultParameterLimit(6,0.01,max_tangential_warp,self.warpset.warp.maxTangentialDistortAmount)
        bounds['p1']=(-2.*maxp1,2.*maxp1)
        maxp2 = self.__findDefaultParameterLimit(7,0.001,max_tangential_warp,self.warpset.warp.maxTangentialDistortAmount)
        bounds['p2']=(-2.*maxp2,2.*maxp2)
        return bounds

    #helper function to find the limit on a parameter that produces the maximum warp
    def __findDefaultParameterLimit(self,parindex,parincrement,warplimit,warpamtfunc) :
        testpars=[self.n/2,self.m/2,MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,0.,0.,0.,0.]
        warpamt=0.; testparval=0.
        while warpamt<warplimit :
            testparval+=parincrement
            testpars[parindex]=testparval
            warpamt=warpamtfunc(testpars)
        return testparval

    #helper function to return the array of the initial population settings for differential_evolution
    def __getInitialPopulation(self,bounds) :
        #initial population is a bunch of linear spaces between the bounds for the parameters individually and in pairs
        init_fit_pars = np.array(self.init_pars)[self.par_mask]
        parnames = np.array(['cx','cy','fx','fy','k1','k2','p1','p2'])[self.par_mask]
        #make a list of each parameter's grid of possible values
        par_variations = []
        nperpar=12-len(parnames)
        for i in range(len(bounds)) :
            name = parnames[i]
            bnds = bounds[i]
            #make the list of this parameter's independent variations to try for the first iteration
            thisparvalues = None
            #principal points and focal lengths are evenly spaced between their bounds
            if name in ['cx','cy','fx','fy'] :
                thisparvalues = np.linspace(bnds[0],bnds[1],nperpar)
            #k1 is evenly spaced between zero and half its max value
            elif name in ['k1'] :
                thisparvalues = np.linspace(0.,0.5*bnds[1],nperpar+1) #+1 because the zero at the beginning is the default parameter value
            #k2 and tangential warping parameters are evenly spaced between +/-(1/2) their bounds
            elif name in ['k2','p1','p2'] :
                thisparvalues = np.linspace(0.5*bnds[0],0.5*bnds[1],nperpar)
            else :
                raise FittingError(f'ERROR: parameter name {name} is not recognized in __getInitialPopulation!')
            par_variations.append(thisparvalues)
        #first member of the population is the nominal initial parameters
        population_list = []
        population_list.append(copy.deepcopy(init_fit_pars))
        #next add a set describing independent parameter variations
        for i in range(len(par_variations)) :
            for val in par_variations[i] :
                if val!=init_fit_pars[i] :
                    toadd = copy.deepcopy(init_fit_pars)
                    toadd[i]=val
                    population_list.append(toadd)
        #add sets describing corner limits of pairs of parameters
        parameter_pair_sets = [np.array([0,0,1,1,2,2,3,3])[self.par_mask], #parameters that have the same effect
                               np.array([0,1,None,None,0,1,None,None])[self.par_mask], #principal points and radial warps 1                                  
                               np.array([0,1,None,None,1,0,None,None])[self.par_mask], #principal points and radial warps 2
                               np.array([None,None,0,1,0,1,None,None])[self.par_mask], #focal lengths and radial warps 1
                               np.array([None,None,0,1,1,0,None,None])[self.par_mask], #focal lengths and radial warps 2
                               np.array([0,1,None,None,None,None,0,1])[self.par_mask], #principal points and tangential warps 1
                               np.array([0,1,None,None,None,None,1,0])[self.par_mask], #principal points and tangential warps 2
                               np.array([None,None,0,1,None,None,0,1])[self.par_mask], #focal lengths and tangential warps 1
                               np.array([None,None,0,1,None,None,1,0])[self.par_mask], #focal lengths and tangential warps 2
                            ]
        for parameter_pair_numbers in parameter_pair_sets :
            pair_numbers_to_consider = set(parameter_pair_numbers)
            if None in pair_numbers_to_consider :
                pair_numbers_to_consider.remove(None)
            for pair_number in pair_numbers_to_consider :
                list_indices = [i for i in range(len(par_variations)) if parameter_pair_numbers[i]==pair_number]
                if len(list_indices)!=2 : 
                    continue
                combinations = [(1,1),(-2,-2),(1,-2),(-2,1)]
                for c in combinations :
                    toadd = copy.deepcopy(init_fit_pars)
                    toadd[list_indices[0]]=par_variations[list_indices[0]][c[0]]
                    toadd[list_indices[1]]=par_variations[list_indices[1]][c[1]]
                    population_list.append(toadd)
        to_return = np.array(population_list)
        logger.info(f'Initial parameter population ({len(population_list)} members):\n{to_return}')
        return to_return

    #helper function to make the constraints
    def __getConstraints(self,fix_k1k2,fix_p1p2,max_radial_warp,max_tangential_warp) :
        constraints = []
        names_to_print = []
        #if k1 and k2 are being fit for, and max_radial_warp is defined, add the max_radial_warp constraint
        if (not fix_k1k2) and max_radial_warp!=-1 :
            constraints.append(scipy.optimize.NonlinearConstraint(
                self._maxRadialDistortAmountForConstraint,
                -1.*max_radial_warp,
                max_radial_warp,
                jac=self._maxRadialDistortAmountJacobianForConstraint,
                hess=scipy.optimize.BFGS()
                )
            )
            names_to_print.append(f'max radial warp={max_radial_warp} pixels')
        #if k1 and k2 are being fit for, and max_tangential_warp is defined, add the max_tangential_warp constraint
        if (not fix_p1p2) and max_tangential_warp!=-1 :
            constraints.append(scipy.optimize.NonlinearConstraint(
                self._maxTangentialDistortAmountForConstraint,
                -1.*max_tangential_warp,
                max_tangential_warp,
                jac=self._maxTangentialDistortAmountJacobianForConstraint,
                hess=scipy.optimize.BFGS()
                )
            )
            names_to_print.append(f'max tangential warp={max_tangential_warp} pixels')
        #print the information about the constraints
        if len(constraints)==0 :
            logger.info('No constraints will be applied')
            return ()
        else :
            constraintstring = 'Will apply constraints: '
            for ntp in names_to_print :
                constraintstring+=ntp+', '
            logger.info(constraintstring[:-2]+'.')
        #return the list of constraints
        if len(constraints)==1 : #if there's only one it doesn't get passed as a list
            return constraints[0]
        else :
            return constraints

    #helper function to get a masking list for the parameters from the warp functions to deal with fixed parameters
    def __getParameterMask(self,fix_cxcy,fix_fxfy,fix_k1k2,fix_p1p2) :
        mask = [True,True,True,True,True,True,True,True]
        if fix_cxcy :
            mask[0]=False
            mask[1]=False
        if fix_fxfy :
            mask[2]=False
            mask[3]=False
        if fix_k1k2 :
            mask[4]=False
            mask[5]=False
        if fix_p1p2 :
            mask[6]=False
            mask[7]=False
        return mask

    #helper function to get a parameter list of the right length so the warp functions always see/return lists of the same length
    def __correctParameterList(self,pars) :
        fixedlist = []; pi=0
        for i,p in enumerate(self.init_pars) :
            if self.par_mask[i] :
                fixedlist.append(pars[pi])
                pi+=1
            else :
                fixedlist.append(p)
        return fixedlist

