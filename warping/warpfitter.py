#imports
from .warpset import WarpSet
from .fitparameterset import FitParameterSet
from .utilities import warp_logger, WarpingError
from .config import CONST
from ..alignment.alignmentset import AlignmentSet
from ..baseclasses.rectangle import rectangleoroverlapfilter
from ..utilities.img_file_io import getImageHWLFromXMLFile
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

    #################### CLASS CONSTANTS ####################

    IM3_EXT = '.im3'                                                  #to replace in filenames read from *_rect.csv

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,samplename,rawfile_top_dir,dbload_top_dir,working_dir,overlaps=-1,layer=1) :
        """
        samplename      = name of the microscope data sample to fit to ("M21_1" or equivalent)
        rawfile_top_dir = path to directory containing [samplename] directory with multilayered ".Data.dat" files in it
        dbload_top_dir  = path to directory containing [samplename]/dbload directory (assuming at least a "rect.csv" and "overlap.csv")
        working_dir     = path to some local directory to store files produced by the WarpFitter
        overlaps        = list of (or two-element tuple of first/last) #s (n) of overlaps to use for evaluating quality of alignment 
                          (default=-1 will use all overlaps)
        layer           = image layer number (indexed starting at 1) to consider in the warping/alignment (default=1)
        """
        #store the directory paths
        self.samp_name = samplename
        self.rawfile_top_dir=rawfile_top_dir
        self.dbload_top_dir=dbload_top_dir
        self.working_dir=working_dir
        #make the alignmentset object to use
        self.bkp_units_mode = units.currentmode
        units.setup("fast") #be sure to use fast units
        self.alignset = self.__initializeAlignmentSet(overlaps=overlaps)
        #get the list of raw file paths
        self.rawfile_paths = [os.path.join(self.rawfile_top_dir,self.samp_name,fn.replace(self.IM3_EXT,CONST.RAW_EXT)) 
                              for fn in [r.file for r in self.alignset.rectangles]]
        #get the size of the images in the sample
        m, n, nlayers = getImageHWLFromXMLFile(self.rawfile_top_dir,samplename)
        if layer<1 or layer>nlayers :
            raise WarpingError(f'ERROR: Choice of layer ({layer}) is not valid for images with {nlayers} layers!')
        #make the warpset object to use
        self.warpset = WarpSet(n=n,m=m,rawfiles=self.rawfile_paths,nlayers=nlayers,layer=layer)
        #for now leave the fitparameter set as None (until the actual fit is called)
        self.fitpars = None

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
        self.warpset.loadRawImages(self.rawfile_paths,self.alignset.overlaps,self.alignset.rectangles,flatfield_file_path,n_threads)
        self.warpset.warpLoadedImages()
        with cd(self.working_dir) :
            if not os.path.isdir(self.samp_name) :
                os.mkdir(self.samp_name)
        self.warpset.writeOutWarpedImages(os.path.join(self.working_dir,self.samp_name))
        self.alignset.getDAPI(filetype='camWarpDAPI',writeimstat=False)

    def doFit(self,fixed=None,normalize=None,float_p1p2_in_polish_fit=False,
              max_radial_warp=10.,max_tangential_warp=10.,p1p2_polish_lasso_lambda=0.,polish=True,
              print_every=1,maxiter=1000,show_plots=False) :
        """
        Fit the cameraWarp model to the loaded dataset
        fixed                    = list of fit parameter names to keep fixed (p1p2 can be fixed separately in the global and polishing minimization steps)
        normalize                = list of fit parameter names to rescale within their bounds for fitting instead of using the raw numbers
        float_p1p2_in_polish_fit = if True, p1 and p2 will not be fixed in the polishing minimization regardless of other arguments
        max_*_warp               = values to use for max warp amount constraints (set to -1 to remove constraints)
        p1p2_polish_lasso_lambda = lambda parameter for LASSO constraint on p1 and p2 during polishing fit
        polish                   = whether to run the polishing fit step at all
        print_every              = print warp parameters and fit results at every [print_every] minimization function calls
        max_iter                 = maximum number of iterations for the global and polishing minimization steps
        """
        #make the set of fit parameters
        self.fitpars = FitParameterSet(fixed,normalize,max_radial_warp,max_tangential_warp,self.warpset.warp)
        #make the iteration counter and the lists of costs/warp amounts
        self.minfunc_calls=0
        self.costs=[]
        self.max_radial_warps=[]
        self.max_tangential_warps=[]
        #silence the AlignmentSet logger
        self.alignset.logger.setLevel(logging.WARN)
        #set the variable describing how often to print progress
        self.print_every = print_every
        #do the global minimization
        minimization_start_time = time.time()
        de_result = self.__runDifferentialEvolution(maxiter)
        init_minimization_done_time = time.time()
        #set the fit parameter values after the global minimization
        self.fitpars.setInitialResults(de_result.x)
        #do the polishing minimization
        if polish :
            result = self.__runPolishMinimization(float_p1p2_in_polish_fit,p1p2_polish_lasso_lambda,maxiter)
        else :
            result=de_result
        polish_minimization_done_time = time.time()
        #set the final fit parameter values
        self.fitpars.setFinalResults(result.x)
        #record the minimization run times
        self.init_min_runtime = init_minimization_done_time-minimization_start_time
        self.polish_min_runtime = polish_minimization_done_time-init_minimization_done_time
        #run all the post-processing stuff
        self.__runPostProcessing(de_result.nfev,show_plots)

    def checkFit(self,fixed=None,normalize=None,float_p1p2_in_polish_fit=False,
                 max_radial_warp=10.,max_tangential_warp=10.,p1p2_polish_lasso_lambda=0.,polish=True) :
        """
        A function to print some information about how the fits will proceed with the current settings
        (see "doFit" function above for what the arguments to this function are)
        """
        #make the set of fit parameters
        self.fitpars = FitParameterSet(fixed,normalize,max_radial_warp,max_tangential_warp,self.warpset.warp)
        #stuff from the differential evolution minimization setup
        warp_logger.info('For global minimization setup:')
        _, _, _ = self.fitpars.getGlobalBoundsConstraintsAndInitialPopulation()
        #stuff for the polishing minimization, if applicable
        if polish :
            warp_logger.info('For polishing minimization setup:')
            _, _ = self.fitpars.getPolishingBoundsAndConstraints(float_p1p2_in_polish_fit)
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
    def __runDifferentialEvolution(self,maxiter) :
        #set up the parameter bounds and constraints and get the initial population
        parameter_bounds, constraints, initial_population = self.fitpars.getGlobalBoundsConstraintsAndInitialPopulation()
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
    def __runPolishMinimization(self,float_p1p2_in_polish_fit,p1p2_polish_lasso_lambda,maxiter) :
        #set up the parameter bounds and constraints
        parameter_bounds, constraints = self.fitpars.getPolishingBoundsAndConstraints(float_p1p2_in_polish_fit)
        #get the indices of the p1 and p2 parameters to lasso if those are floating
        lasso_indices = None
        if float_p1p2 and lasso_lambda!=0. :
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

    # helper function to create and return a new alignmentSet object that's set up to run on the identified set of images/overlaps
    def __initializeAlignmentSet(self, *, overlaps) :
        #If this is running on my Mac I want to be asked which GPU device to use because it doesn't default to the AMD compute unit....
        customGPUdevice = True if platform.system()=='Darwin' else False
        a = AlignmentSet(self.dbload_top_dir,self.working_dir,self.samp_name,interactive=customGPUdevice,useGPU=True,
                         selectoverlaps=rectangleoroverlapfilter(overlaps, compatibility=True),onlyrectanglesinoverlaps=True)
        return a

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
    def __runPostProcessing(self,n_initial_fevs,show_plots) :
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
