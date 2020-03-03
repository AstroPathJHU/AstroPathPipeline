#imports
from .warpset import WarpSet
from .alignmentset import AlignmentSet, Rectangle
from .overlap import Overlap
from .tableio import readtable,writetable
import numpy as np, scipy, matplotlib.pyplot as plt
import os, logging, copy

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
    def __init__(self,samplename,rawfile_dir,metafile_dir,working_dir,overlaps=-1,layers=[1],warpset=None,warp=None) :
        """
        samplename   = name of the microscope data sample to fit to ("M21_1" or equivalent)
        rawfile_dir  = path to directory containing multilayered ".raw" files
        metafile_dir = path to directory containing "dbload" metadata files (assuming at least a "rect.csv" and "overlap.csv")
        working_dir  = path to some local directory to store files produced by the WarpFitter
        overlaps     = list of (or two-element tuple of first/last) #s (n) of overlaps to use for evaluating quality of alignment 
                       (default=-1 will use all overlaps)
        layers       = list of image layer numbers (indexed starting at 1) to consider in the warping/alignment (default=[1])
        warpset      = WarpSet object to initialize with (optional, a new WarpSet will be created if None) 
        warp         = CameraWarp object whose optimal parameters will be determined (optional, if None a new one will be created)
        """
        #store the directory paths
        self.samp_name = samplename
        self.rawfile_dir=rawfile_dir
        self.metafile_dir=metafile_dir
        self.working_dir=working_dir
        self.init_dir = os.getcwd()
        #setup the working directory and the lists of overlaps and rectangles
        self.overlaps, self.rectangles = self.__setupWorkingDirectory(overlaps)
        #get the list of raw file paths
        self.rawfile_paths = [os.path.join(self.rawfile_dir,fn.replace(IM3_EXT,RAW_EXT)) for fn in [r.file for r in self.rectangles]]
        #get the size of the images in the sample
        self.n, self.m, self.nlayers = self.__getImageSizesFromImmFile()
        for layer in layers :
            if layer<1 or layer>self.nlayers :
                raise FittingError(f'Choice of layers ({layers}) is not valid for images with {self.nlayers} layers!')
        #make the warpset object to use
        if warpset is not None :
            self.warpset = warpset
        elif warp is not None :
            if warp.n!=self.n or warp.m!=self.m :
                msg = f'Warp object passed to WarpFitter is set to run on images of size ({warp.n},{warp.m}),'
                msg+=f' not of size ({self.n},{self.m}) as specified by .imm files'
                raise FittingError(msg)
            self.warpset = WarpSet(warp=warp,rawfiles=self.rawfile_paths,nlayers=self.nlayers,layers=layers)
        else :
            self.warpset = WarpSet(n=self.n,m=self.m,rawfiles=self.rawfile_paths,nlayers=self.nlayers,layers=layers)
        #make the alignmentset object to use
        self.alignset = self.__initializeAlignmentSet()
        #the private variable that will hold the best-fit warp
        self.__best_fit_warp = None

    #################### PUBLIC FUNCTIONS ####################

    def loadRawFiles(self) :
        """
        Load the raw files into the warpset, warp/save them, and load them into the alignment set 
        """
        self.warpset.loadRawImageSet(self.rawfile_paths)
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
        self.alignset.getDAPI(filetype='camWarpDAPI')

    def doFit(self,fix_cxcy=False,fix_fxfy=False,fix_k1k2=False,fix_p1p2=False,max_radial_warp=25.,max_tangential_warp=25.,par_bounds=None,print_every=1,show_plots=False) :
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
        #get the list of constraints
        constraints = self.__getConstraints(fix_k1k2,fix_p1p2,max_radial_warp,max_tangential_warp)
        #get the list to use to mask fixed parameters in the minimization functions
        self.par_mask = self.__getParameterMask(fix_cxcy,fix_fxfy,fix_k1k2,fix_p1p2)
        #get the list of initial parameters to copy from when necessary
        self.init_pars = self.warpset.getListOfWarpParameters()
        #set the variable describing how often to print progress
        self.print_every = print_every
        #call differential_evolution
        logger.info('Starting minimization....')
        os.chdir(self.working_dir)
        try :
            result=scipy.optimize.differential_evolution(
                self._evalCamWarpOnAlignmentSet,
                parameter_bounds,
                constraints=constraints
                )
        except Exception :
            raise FittingError('Something failed in the minimization!')
        finally :
            os.chdir(self.init_dir)
        logger.info(f'Minimization completed {"successfully" if result.success else "UNSUCCESSFULLY"} in {result.nfev} evaluations.')
        #make the fit progress plots
        self.__makeFitProgressPlots(show_plots)
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
        self.warpset.warpLoadedImageSet()
        #reload the (newly-warped) images into the alignment set
        self.alignset.updateRectangleImages(self.warpset.warped_images,'.raw')
        #align the images 
        cost = self.alignset.align()
        #add to the lists to plot
        self.costs.append(cost if cost<1e10 else -999)
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
    def __makeFitProgressPlots(self,show) :
        iters = np.linspace(0,len(self.costs),len(self.costs),endpoint=False)
        f,(ax1,ax2,ax3) = plt.subplots(1,3)
        f.set_size_inches(20.,5.)
        ax1.plot(iters,self.costs)
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('cost')
        ax2.plot(iters,self.max_radial_warps)
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('max radial warp')
        ax3.plot(iters,self.max_tangential_warps)
        ax3.set_xlabel('iteration')
        ax3.set_ylabel('max tangential warp')
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
        #make sure the best fit warp exists (which means the warpset is updated with the best fit parameters)
        if self.__best_fit_warp is None :
            raise FittingError('Do not call __makeBestFitAlignmentComparisonImages until after the best fit warp has been set!')
        self.warpset.warpLoadedImageSet()
        #reload the (newly-warped) images into the alignment set
        self.alignset.updateRectangleImages(self.warpset.warped_images,'.raw')
        #align the images 
        bestcost = self.alignset.align()
        #write out the overlap comparison figures
        figure_dir_name = 'alignment_overlap_comparisons'
        os.chdir(self.working_dir)
        if not os.path.isdir(figure_dir_name) :
            os.mkdir(figure_dir_name)
        os.chdir(figure_dir_name)
        try :
            self.alignset.writeOverlapComparisonImages()
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
        a.overlaps=self.overlaps
        a.rectangles=self.rectangles
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
        # cx/cy bounds are +/- 35% of the center point
        bounds['cx']=(0.65*(self.n/2.),1.35*(self.n/2.))
        bounds['cy']=(0.65*(self.m/2.),1.35*(self.m/2.))
        # fx/fy bounds are +/- 2% of the nominal values 
        bounds['fx']=(0.98*MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
        bounds['fy']=(0.98*MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
        # k1/k2 and p1/p2 bounds are twice those that would produce the max radial and tangential warp, respectively, with all others zero
        maxk1 = self.__findDefaultParameterLimit(4,1,max_radial_warp,self.warpset.warp.maxRadialDistortAmount)
        bounds['k1']=(-2.*maxk1,2.*maxk1)
        maxk2 = self.__findDefaultParameterLimit(5,1000,max_radial_warp,self.warpset.warp.maxRadialDistortAmount)
        bounds['k2']=(-2.*maxk2,2.*maxk2)
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

