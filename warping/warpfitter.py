#imports
from .warpset import WarpSet
from .utilities import warp_logger, WarpingError
from .config import CONST
from ..alignment.alignmentset import AlignmentSet
from ..baseclasses.rectangle import rectangleoroverlapfilter
from ..utilities.img_file_io import getImageHWLFromXMLFile, getRawAsHWL
from ..utilities import units
from ..utilities.misc import cd
import numpy as np, scipy, matplotlib.pyplot as plt
import os, copy, shutil, platform, time, logging

class WarpFitter :
    """
    Main class for fitting a camera matrix and distortion parameters to a set of images based on the results of their alignment
    """

    #################### PROPERTIES ####################

    @property
    def m(self) :
        return self.warpset.warp.m #image height
    @property
    def n(self) :
        return self.warpset.warp.n #image width
    @property
    def nlayers(self) :
        return self.warpset.nlayers #number of image layers
    @property
    def layer(self) :
        return self.warpset.layer # which layer is being used to fit the warping

    #################### CLASS CONSTANTS ####################

    IM3_EXT = '.im3'                                                  #to replace in filenames read from *_rect.csv
    MICROSCOPE_OBJECTIVE_FOCAL_LENGTH = 40000.                        #focal length of the microscope objective (20mm) in pixels
    FITPAR_NAME_LIST = ['cx','cy','fx','fy','k1','k2','p1','p2','k3'] #ordered list of fit parameters names

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,samplename,rawfile_top_dir,dbload_top_dir,working_dir,overlaps=-1,layer=1,warpset=None,warp=None) :
        """
        samplename      = name of the microscope data sample to fit to ("M21_1" or equivalent)
        rawfile_top_dir = path to directory containing [samplename] directory with multilayered ".Data.dat" files in it
        dbload_top_dir  = path to directory containing [samplename]/dbload directory (assuming at least a "rect.csv" and "overlap.csv")
        working_dir     = path to some local directory to store files produced by the WarpFitter
        overlaps        = list of (or two-element tuple of first/last) #s (n) of overlaps to use for evaluating quality of alignment 
                          (default=-1 will use all overlaps)
        layer           = image layer number (indexed starting at 1) to consider in the warping/alignment (default=1)
        warpset         = WarpSet object to initialize with (optional, a new WarpSet will be created if None) 
        warp            = CameraWarp object whose optimal parameters will be determined (optional, if None a new one will be created)
        """
        #store the directory paths
        self.samp_name = samplename
        self.rawfile_top_dir=rawfile_top_dir
        self.rawfile_dir=os.path.join(self.rawfile_top_dir,self.samp_name)
        self.dbload_top_dir=dbload_top_dir
        self.metafile_dir=os.path.join(self.dbload_top_dir,self.samp_name,'dbload')
        self.working_dir=working_dir
        #make sure we're using fast units before making the alignmentset
        self.bkp_units_mode = units.currentmode
        units.setup("fast")
        #make the alignmentset object to use
        self.alignset = self.__initializeAlignmentSet(overlaps=overlaps)
        self.rectangles = self.alignset.rectangles
        self.overlaps = self.alignset.overlaps
        #setup the working directory and the lists of overlaps and rectangles
        self.__setupWorkingDirectory()
        #get the list of raw file paths
        self.rawfile_paths = [os.path.join(self.rawfile_dir,fn.replace(self.IM3_EXT,CONST.RAW_EXT)) for fn in [r.file for r in self.rectangles]]
        #get the size of the images in the sample
        m, n, nlayers = getImageHWLFromXMLFile(self.rawfile_top_dir,samplename)
        if layer<1 or layer>nlayers :
            raise WarpingError(f'Choice of layer ({layer}) is not valid for images with {nlayers} layers!')
        #make the warpset object to use
        if warpset is not None :
            self.warpset = warpset
        elif warp is not None :
            if warp.n!=n or warp.m!=m :
                msg = f'Warp object passed to WarpFitter is set to run on images of size ({warp.n},{warp.m}),'
                msg+=f' not of size ({n},{m}) as specified by .xml files'
                raise WarpingError(msg)
            self.warpset = WarpSet(warp=warp,rawfiles=self.rawfile_paths,nlayers=nlayers,layer=layer)
        else :
            self.warpset = WarpSet(n=n,m=m,rawfiles=self.rawfile_paths,nlayers=nlayers,layer=layer)
        #the private variable that will hold the best-fit warp
        self.__best_fit_warp = None

    def __del__(self) :
        """
        Remove the placeholder files when the object is being deleted
        """
        if os.path.isdir(os.path.join(self.working_dir,self.samp_name)) :
            warp_logger.info('Removing copied raw layer files....')
            with cd(self.working_dir) :
                shutil.rmtree(self.samp_name)
        try:
            units.setup(self.bkp_units_mode)
        except TypeError: #units was garbage collected before the warpfitter
            pass

    #################### PUBLIC FUNCTIONS ####################

    def loadRawFiles(self,flatfield_file_path=None,n_threads=1) :
        """
        Load the raw files into the warpset, warp/save them, and load them into the alignment set 
        flatfield_file_path = path to the flatfield file to use in correcting the rawfile illumination
        n_threads           = how many different processes to run when loading files
        """
        warp_logger.info(f'Loading flatfield file {flatfield_file_path} to correct raw image illumination')
        if flatfield_file_path is not None :
            flatfield_file_layer = (getRawAsHWL(flatfield_file_path,self.m,self.n,self.nlayers,np.float64))[:,:,self.layer-1] 
        else :
            flatfield_file_layer = np.ones((self.m,self.n),dtype=np.float64)
        self.warpset.loadRawImages(self.rawfile_paths,self.overlaps,self.rectangles,flatfield_file_layer,n_threads)
        self.warpset.warpLoadedImages()
        with cd(self.working_dir) :
            if not os.path.isdir(self.samp_name) :
                os.mkdir(self.samp_name)
        self.warpset.writeOutWarpedImages(os.path.join(self.working_dir,self.samp_name))
        self.alignset.getDAPI(filetype='camWarpDAPI',writeimstat=False)

    def doFit(self,fix_cxcy=False,fix_fxfy=False,fix_k1k2k3=False,fix_p1p2_in_global_fit=False,fix_p1p2_in_polish_fit=False,
              max_radial_warp=10.,max_tangential_warp=10.,p1p2_polish_lasso_lambda=0.,polish=True,print_every=1,maxiter=1000,show_plots=False) :
        """
        Fit the cameraWarp model to the loaded dataset
        fix_*                    = set True to fix groups of parameters (p1p2 can be fixed separately in the global and polishing minimization steps)
        max_*_warp               = values to use for max warp amount constraints (set to -1 to remove constraints)
        p1p2_polish_lasso_lambda = lambda parameter for LASSO constraint on p1 and p2 during polishing fit
        polish                   = whether to run the polishing fit step at all
        print_every              = print warp parameters and fit results at every [print_every] minimization function calls
        max_iter                 = maximum number of iterations for the global and polishing minimization steps
        """
        #make the iteration counter and the lists of costs/warp amounts
        self.minfunc_calls=0
        self.costs=[]
        self.max_radial_warps=[]
        self.max_tangential_warps=[]
        #silence the AlignmentSet logger
        self.alignset.logger.setLevel(logging.WARN)
        #get the list of initial parameters to copy from when necessary
        self.init_pars = self.warpset.getListOfWarpParameters()
        #figure out the default bounds to reference when needed
        self.default_bounds = self.__buildDefaultParameterBoundsDict(max_radial_warp,max_tangential_warp)
        #set the variable describing how often to print progress
        self.print_every = print_every
        #do the global minimization
        minimization_start_time = time.time()
        de_result = self.__runDifferentialEvolution(fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2_in_global_fit,max_radial_warp,max_tangential_warp,maxiter)
        init_minimization_done_time = time.time()
        #do the polishing minimization
        if polish :
            init_pars = self.__correctParameterList(de_result.x)
            result = self.__runPolishMinimization(fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2_in_polish_fit,max_radial_warp,max_tangential_warp,p1p2_polish_lasso_lambda,init_pars,maxiter)
        else :
            result=de_result
        polish_minimization_done_time = time.time()
        #record the minimization run times
        self.init_min_runtime = init_minimization_done_time-minimization_start_time
        self.polish_min_runtime = polish_minimization_done_time-init_minimization_done_time
        #run all the post-processing stuff
        self.__runPostProcessing(result,de_result.nfev,show_plots)
        return result

    def checkFit(self,fix_cxcy=False,fix_fxfy=False,fix_k1k2k3=False,fix_p1p2_in_global_fit=False,fix_p1p2_in_polish_fit=False,
              max_radial_warp=10.,max_tangential_warp=10.,p1p2_polish_lasso_lambda=0.,polish=True) :
        """
        A function to print some information about how the fits will proceed with the current settings
        (see "doFit" function above for what the arguments to this function are)
        """
        #figure out the default bounds
        self.default_bounds = self.__buildDefaultParameterBoundsDict(max_radial_warp,max_tangential_warp)
        #stuff from the differential evolution minimization setup
        warp_logger.info('For global minimization setup:')
        _, _ = self.__getParameterBoundsAndConstraints(fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2_in_global_fit,max_radial_warp,max_tangential_warp)
        #stuff for the polishing minimization, if applicable
        if polish :
            warp_logger.info('For polishing minimization setup:')
            _, _ = self.__getParameterBoundsAndConstraints(fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2_in_polish_fit,max_radial_warp,max_tangential_warp)
            if p1p2_polish_lasso_lambda!=0 :
                warp_logger.info(f'p1 and p2 will be LASSOed in the polishing minimization with lambda = {p1p2_polish_lasso_lambda}')
            else :
                warp_logger.info('p1 and p2 will not be LASSOed in the polishing minimization')
        else :
            warp_logger.info('Polishing minimization will be skipped')

    #################### FUNCTIONS FOR USE WITH MINIMIZATION ####################

    # !!!!!! For the time being, these functions don't correctly describe dependence on k4, k5, or k6 !!!!!!

    #The function whose return value is minimized by the fitting
    def _evalCamWarpOnAlignmentSet(self,pars,max_rad_warp,max_tan_warp,lasso_param_indices,lasso_lambda) :
        self.minfunc_calls+=1
        #first fix the parameter list so the warp functions always see vectors of the same length
        fixedpars = self.__correctParameterList(pars)
        #update the warp with the new parameters
        self.warpset.updateCameraParams(fixedpars)
        #then warp the images
        self.warpset.warpLoadedImages(skip_corners=self.skip_corners)
        #reload the (newly-warped) images into the alignment set
        self.alignset.updateRectangleImages([warpimg for warpimg in self.warpset.images if not (self.skip_corners and warpimg.is_corner_only)])
        #check the warp amounts to see if the sample should be realigned
        rad_warp = self.warpset.warp.maxRadialDistortAmount(fixedpars)
        tan_warp = self.warpset.warp.maxTangentialDistortAmount(fixedpars)
        align_strategy = 'overwrite' if (abs(rad_warp)<max_rad_warp or max_rad_warp==-1) and (tan_warp<max_tan_warp or max_tan_warp==-1) else 'shift_only'
        #align the images and get the cost
        aligncost = self.alignset.align(skip_corners=self.skip_corners,
                                   write_result=False,
                                   return_on_invalid_result=True,
                                   alreadyalignedstrategy=align_strategy,
                                   warpwarnings=True,
                                   )
        #compute the cost from the LASSO constraint on the specified parameters
        lasso_cost=0.
        if lasso_param_indices is not None :
            for pindex in lasso_param_indices :
                lasso_cost += lasso_lambda*abs(pars[pindex])
        #sum the costs
        cost = aligncost+lasso_cost
        #add to the lists to plot
        self.costs.append(cost if cost<1e10 else -0.1)
        self.max_radial_warps.append(rad_warp)
        self.max_tangential_warps.append(tan_warp)
        #print progress if requested
        if self.minfunc_calls%self.print_every==0 :
            warp_logger.info(self.warpset.warp.paramString())
            msg = f'  Call {self.minfunc_calls} cost={cost:.04f}' 
            if lasso_param_indices is not None : msg+=f'(={aligncost:.04f}+{lasso_cost:.04f})'
            msg+=f' (radial warp={self.max_radial_warps[-1]:.02f}, tangential warp={self.max_tangential_warps[-1]:.02f})'
            warp_logger.info(msg)
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

    #function to run global minimization with differential evolution and return the result
    def __runDifferentialEvolution(self,fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2,max_radial_warp,max_tangential_warp,maxiter) :
        #set up the parameter bounds and constraints
        parameter_bounds, constraints = self.__getParameterBoundsAndConstraints(fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2,max_radial_warp,max_tangential_warp)
        #get the array of the initial population
        initial_population=self.__getInitialPopulation(parameter_bounds)
        #run the minimization
        warp_logger.info('Starting initial minimization....')
        self.skip_corners = True
        with cd(self.working_dir) :
            try :
                result=scipy.optimize.differential_evolution(
                    func=self._evalCamWarpOnAlignmentSet,
                    bounds=parameter_bounds,
                    args=(max_radial_warp,max_tangential_warp,None,0.),
                    strategy='best2bin',
                    maxiter=maxiter,
                    tol=0.03,
                    mutation=(0.6,1.00),
                    recombination=0.7,
                    polish=False,
                    init=initial_population,
                    constraints=constraints
                    )
            except Exception :
                raise WarpingError('Something failed in the initial minimization!')
        warp_logger.info(f'Initial minimization completed {"successfully" if result.success else "UNSUCCESSFULLY"} in {result.nfev} evaluations.')
        return result

    #function to run local polishing minimiation with trust-constr and return the result
    def __runPolishMinimization(self,fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2,max_radial_warp,max_tangential_warp,lasso_lambda,init_pars,maxiter) :
        #set up the parameter bounds and constraints
        parameter_bounds, constraints = self.__getParameterBoundsAndConstraints(fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2,max_radial_warp,max_tangential_warp)
        #get the indices of the p1 and p2 parameters to lasso if those are floating
        lasso_indices = None
        if (not fix_p1p2) and lasso_lambda!=0. :
            relevant_parnamelist = (np.array(self.FITPAR_NAME_LIST)[self.par_mask]).tolist()
            lasso_indices = (relevant_parnamelist.index('p1'),relevant_parnamelist.index('p2'))
        #figure out the initial parameters and their step sizes
        init_pars_to_use = ((np.array(init_pars))[self.par_mask]).tolist()
        for i in range(len(init_pars_to_use)) :
            if init_pars_to_use[i]==0. :
                init_pars_to_use[i]=0.00000001 # can't send p1/p2=0. to the Jacobian functions in trust-constr
        relative_steps = np.array([abs(0.05*p) if abs(p)<1. else 0.05 for p in init_pars_to_use])
        #call minimize with trust_constr
        warp_logger.info('Starting polishing minimization....')
        with cd(self.working_dir) :
            try :
                result=scipy.optimize.minimize(
                    fun=self._evalCamWarpOnAlignmentSet,
                    x0=init_pars_to_use,
                    args=(max_radial_warp,max_tangential_warp,lasso_indices,lasso_lambda),
                    method='trust-constr',
                    bounds=parameter_bounds,
                    constraints=constraints,
                    options={'xtol':1e-4,'gtol':1e-5,'finite_diff_rel_step':relative_steps,'maxiter':maxiter}
                    )
            except Exception :
                raise WarpingError('Something failed in the polishing minimization!')
        msg = f'Final minimization completed {"successfully" if result.success else "UNSUCCESSFULLY"} in {result.nfev} evaluations '
        term_conds = {0:'max iterations',1:'gradient tolerance',2:'parameter tolerance',3:'callback function'}
        msg+=f'due to compliance with {term_conds[result.status]} criteria.'
        warp_logger.info(msg)
        return result

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
        with cd(self.working_dir) :
            try :
                plt.savefig('fit_progress.png')
                if show :
                    plt.show()
                plt.close()
            except Exception :
                raise WarpingError('something went wrong while trying to save the fit progress plots!')
        return ninitev, len(self.costs)-ninitev

    #function to save alignment comparison visualizations in a new directory inside the working directory
    def __makeBestFitAlignmentComparisonImages(self) :
        warp_logger.info('writing out warping/alignment comparison images')
        #make sure the best fit warp exists (which means the warpset is updated with the best fit parameters)
        if self.__best_fit_warp is None :
            raise WarpingError('Do not call __makeBestFitAlignmentComparisonImages until after the best fit warp has been set!')
        #start by aligning the raw, unwarped images and getting their shift comparison information/images
        self.alignset.updateRectangleImages(self.warpset.images,usewarpedimages=False)
        rawcost = self.alignset.align(write_result=False,alreadyalignedstrategy="overwrite",warpwarnings=True)
        raw_overlap_comparisons_dict = self.alignset.getOverlapComparisonImagesDict()
        #next warp and align the images with the best fit warp
        self.warpset.warpLoadedImages()
        self.alignset.updateRectangleImages(self.warpset.images)
        bestcost = self.alignset.align(write_result=False,alreadyalignedstrategy="overwrite",warpwarnings=True)
        warped_overlap_comparisons_dict = self.alignset.getOverlapComparisonImagesDict()
        warp_logger.info(f'Alignment cost from raw images = {rawcost:.08f}; alignment cost from warped images = {bestcost:.08f} ({(100*(1.-bestcost/rawcost)):.04f}% reduction)')
        #write out the overlap comparison figures
        figure_dir_name = 'alignment_overlap_comparisons'
        with cd(self.working_dir) :
            if not os.path.isdir(figure_dir_name) :
                os.mkdir(figure_dir_name)
            with cd(figure_dir_name) :
                try :
                    for overlap_identifier in raw_overlap_comparisons_dict.keys() :
                        code = overlap_identifier[0]
                        fn   = overlap_identifier[1]
                        pix_to_in = 20./self.n
                        if code in [2,8] :
                            f,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
                            f.set_size_inches(self.n*pix_to_in,4.5*0.2*self.m*pix_to_in)
                            order = [ax1,ax3,ax2,ax4]
                        elif code in [1,3,7,9] :
                            f,ax = plt.subplots(2,2)
                            f.set_size_inches(2.*0.2*self.n*pix_to_in,2*0.2*self.m*pix_to_in)
                            order = [ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
                        elif code in [4,6] :
                            f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,sharey=True)
                            f.set_size_inches(4.5*0.2*self.n*pix_to_in,self.m*pix_to_in)
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
                        plt.close()
                except Exception :
                    raise WarpingError('Something went wrong while trying to write out the overlap comparison images')
        return rawcost, bestcost

    #################### PRIVATE HELPER FUNCTIONS ####################

    # helper function to create the working directory and create/write out lists of overlaps and rectangles
    def __setupWorkingDirectory(self) :
        if not os.path.isdir(self.working_dir) :
            os.mkdir(self.working_dir)

    # helper function to create and return a new alignmentSet object that's set up to run on the identified set of images/overlaps
    def __initializeAlignmentSet(self, *, overlaps) :
        #If this is running on my Mac I want to be asked which GPU device to use because it doesn't default to the AMD compute unit....
        customGPUdevice = True if platform.system()=='Darwin' else False
        a = AlignmentSet(self.dbload_top_dir,self.working_dir,self.samp_name,interactive=customGPUdevice,useGPU=True,selectoverlaps=rectangleoroverlapfilter(overlaps, compatibility=True),onlyrectanglesinoverlaps=True)
        return a

    #helper function to create and return the lists of fit parameter bounds and fit constraints
    def __getParameterBoundsAndConstraints(self,fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2,max_radial_warp,max_tangential_warp) :
        #build the list of parameter bounds
        parameter_bounds = self.__getParameterBoundsList(fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2)
        warp_logger.info(f'Floating parameter bounds = {parameter_bounds}')
        #get the list to use to mask fixed parameters in the minimization functions
        self.par_mask = self.__getParameterMask(fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2)
        #get the list of constraints
        constraints = self.__getConstraints(fix_k1k2k3,fix_p1p2,max_radial_warp,max_tangential_warp)
        return parameter_bounds, constraints

    #helper function to make the list of parameter bounds for fitting
    def __getParameterBoundsList(self,fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2) :
        #copy the whole set of default
        bounds_dict = copy.deepcopy(self.default_bounds)
        #remove any parameters that will be fixed
        to_remove = []
        if fix_cxcy :
            to_remove+=['cx','cy']
        if fix_fxfy :
            to_remove+=['fx','fy']
        if fix_k1k2k3 :
            to_remove+=['k1','k2','k3']
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
        warp_logger.info(msg)
        #return the ordered list of parameters
        return [bounds_dict[name] for name in self.FITPAR_NAME_LIST if name in bounds_dict.keys()]

    #helper function to make the default list of parameter constraints
    def __buildDefaultParameterBoundsDict(self,max_radial_warp,max_tangential_warp) :
        bounds = {}
        # cx/cy bounds are +/- 10% of the center point
        bounds['cx']=(0.8*(self.n/2.),1.2*(self.n/2.))
        bounds['cy']=(0.8*(self.m/2.),1.2*(self.m/2.))
        # fx/fy bounds are +/- 2% of the nominal values 
        bounds['fx']=(0.98*self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
        bounds['fy']=(0.98*self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
        # k1/k2/k3 and p1/p2 bounds are 1.5x those that would produce the max radial and tangential warp, respectively, with all others zero
        # (except k1 can't be negative)
        maxk1 = self.__findDefaultParameterLimit(4,0.1,max_radial_warp,self.warpset.warp.maxRadialDistortAmount)
        bounds['k1']=(0.,1.5*maxk1)
        maxk2 = self.__findDefaultParameterLimit(5,100,max_radial_warp,self.warpset.warp.maxRadialDistortAmount)
        bounds['k2']=(-1.5*maxk2,1.5*maxk2)
        maxk3 = self.__findDefaultParameterLimit(8,10000,max_radial_warp,self.warpset.warp.maxRadialDistortAmount)
        bounds['k3']=(-1.5*maxk3,1.5*maxk3)
        maxp1 = self.__findDefaultParameterLimit(6,0.001,max_tangential_warp,self.warpset.warp.maxTangentialDistortAmount)
        bounds['p1']=(-1.5*maxp1,1.5*maxp1)
        maxp2 = self.__findDefaultParameterLimit(7,0.001,max_tangential_warp,self.warpset.warp.maxTangentialDistortAmount)
        bounds['p2']=(-1.5*maxp2,1.5*maxp2)
        return bounds

    #helper function to find the limit on a parameter that produces the maximum warp
    def __findDefaultParameterLimit(self,parindex,parincrement,warplimit,warpamtfunc) :
        testpars=[self.n/2,self.m/2,self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,0.,0.,0.,0.,0.]
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
        parnames = np.array(self.FITPAR_NAME_LIST)[self.par_mask]
        #make a list of each parameter's grid of possible values
        par_variations = []
        nperpar=5
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
            #k2, k3, and tangential warping parameters are evenly spaced between +/-(1/2) their bounds
            elif name in ['k2','k3','p1','p2'] :
                thisparvalues = np.linspace(0.5*bnds[0],0.5*bnds[1],nperpar)
            else :
                raise WarpingError(f'ERROR: parameter name {name} is not recognized in __getInitialPopulation!')
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
        #add sets describing corner limits of groups of parameters
        parameter_group_sets = [np.array([0,0,1,1,2,2,3,3,2])[self.par_mask], #parameters that have the same effect
                                np.array([0,1,None,None,0,1,None,None,0])[self.par_mask], #principal points and radial warps 1                                  
                                np.array([0,1,None,None,1,0,None,None,1])[self.par_mask], #principal points and radial warps 2
                                np.array([0,1,None,None,0,1,None,None,1])[self.par_mask], #principal points and radial warps 3                                  
                                np.array([0,1,None,None,1,0,None,None,0])[self.par_mask], #principal points and radial warps 4
                                np.array([None,None,0,1,0,1,None,None,0])[self.par_mask], #focal lengths and radial warps 1
                                np.array([None,None,0,1,1,0,None,None,1])[self.par_mask], #focal lengths and radial warps 2
                                np.array([None,None,0,1,0,1,None,None,1])[self.par_mask], #focal lengths and radial warps 3
                                np.array([None,None,0,1,1,0,None,None,0])[self.par_mask], #focal lengths and radial warps 4
                                np.array([0,1,None,None,None,None,0,1,None])[self.par_mask], #principal points and tangential warps 1
                                np.array([0,1,None,None,None,None,1,0,None])[self.par_mask], #principal points and tangential warps 2
                                np.array([None,None,0,1,None,None,0,1,None])[self.par_mask], #focal lengths and tangential warps 1
                                np.array([None,None,0,1,None,None,1,0,None])[self.par_mask], #focal lengths and tangential warps 2
                                ]
        for parameter_group_numbers in parameter_group_sets :
            group_numbers_to_consider = set(parameter_group_numbers)
            if None in group_numbers_to_consider :
                group_numbers_to_consider.remove(None)
            for group_number in group_numbers_to_consider :
                list_indices = [i for i in range(len(par_variations)) if parameter_group_numbers[i]==group_number]
                combinations = []
                if len(list_indices)==2 :
                    combinations = [(1,1),(-2,-2),(1,-2),(-2,1),(0,0),(-1,-1),(0,-1),(-1,0)]
                elif len(list_indices)==3 :
                    combinations = [(1,1,1),(-2,-2,1),(1,-2,1),(-2,1,1),(1,1,-2),(-2,-2,-2),(1,-2,-2),(-2,1,-2)]
                for c in combinations :
                    toadd = copy.deepcopy(init_fit_pars)
                    toadd[list_indices[0]]=0.2*par_variations[list_indices[0]][c[0]]
                    toadd[list_indices[1]]=0.2*par_variations[list_indices[1]][c[1]]
                    if len(list_indices)==3 :
                        toadd[list_indices[2]]=0.2*par_variations[list_indices[2]][c[2]]
                    population_list.append(toadd)
        to_return = np.array(population_list)
        warp_logger.info(f'Initial parameter population ({len(population_list)} members):\n{to_return}')
        return to_return

    #helper function to make the constraints
    def __getConstraints(self,fix_k1k2k3,fix_p1p2,max_radial_warp,max_tangential_warp) :
        constraints = []
        names_to_print = []
        #if k1 and k2 are being fit for, and max_radial_warp is defined, add the max_radial_warp constraint
        if (not fix_k1k2k3) and max_radial_warp!=-1 :
            constraints.append(scipy.optimize.NonlinearConstraint(
                self._maxRadialDistortAmountForConstraint,
                -1.*max_radial_warp,
                max_radial_warp,
                jac=self._maxRadialDistortAmountJacobianForConstraint,
                hess=scipy.optimize.BFGS()
                )
            )
            names_to_print.append(f'max radial warp={max_radial_warp} pixels')
        #if p1 and p2 are being fit for, and max_tangential_warp is defined, add the max_tangential_warp constraint
        if (not fix_p1p2) and max_tangential_warp!=-1 :
            constraints.append(scipy.optimize.NonlinearConstraint(
                self._maxTangentialDistortAmountForConstraint,
                0.,
                max_tangential_warp,
                jac=self._maxTangentialDistortAmountJacobianForConstraint,
                hess=scipy.optimize.BFGS()
                )
            )
            names_to_print.append(f'max tangential warp={max_tangential_warp} pixels')
        #print the information about the constraints
        if len(constraints)==0 :
            warp_logger.info('No constraints will be applied')
            return ()
        else :
            constraintstring = 'Will apply constraints: '
            for ntp in names_to_print :
                constraintstring+=ntp+', '
            warp_logger.info(constraintstring[:-2]+'.')
        #return the list of constraints
        if len(constraints)==1 : #if there's only one it doesn't get passed as a list
            return constraints[0]
        else :
            return constraints

    #helper function to get a masking list for the parameters from the warp functions to deal with fixed parameters
    def __getParameterMask(self,fix_cxcy,fix_fxfy,fix_k1k2k3,fix_p1p2) :
        mask = [True,True,True,True,True,True,True,True,True]
        if fix_cxcy :
            mask[0]=False
            mask[1]=False
        if fix_fxfy :
            mask[2]=False
            mask[3]=False
        if fix_k1k2k3 :
            mask[4]=False
            mask[5]=False
            mask[8]=False
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

    #helper function to run all of the various post-processing functions after the fit is done
    def __runPostProcessing(self,fitresult,n_initial_fevs,show_plots) :
        #make the fit progress plots
        self.init_its, self.polish_its = self.__makeFitProgressPlots(n_initial_fevs,show_plots)
        #use the fit result to make the best fit warp object
        best_fit_pars = self.__correctParameterList(fitresult.x)
        self.warpset.updateCameraParams(best_fit_pars)
        self.__best_fit_warp = copy.deepcopy(self.warpset.warp)
        warp_logger.info('Best fit parameters:')
        self.__best_fit_warp.printParams()
        #write out the set of alignment comparison images
        self.raw_cost, self.best_cost = self.__makeBestFitAlignmentComparisonImages()
        #save the figure of the best-fit warp fields
        with cd(self.working_dir) :
            self.__best_fit_warp.makeWarpAmountFigure()
            self.__best_fit_warp.writeParameterTextFile(self.par_mask,
                                                        self.init_its,self.polish_its,
                                                        self.init_min_runtime,self.polish_min_runtime,
                                                        self.raw_cost,self.best_cost)
