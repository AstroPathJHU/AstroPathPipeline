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
import os, copy, math, shutil, platform, time, logging

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

    IM3_EXT = '.im3'                                          #to replace in filenames read from *_rect.csv
    DE_TOLERANCE = 0.03                                       #tolerance for the differential evolution minimization
    DE_MUTATION = (0.6,1.00)                                  #mutation bounds for differential evolution minimization
    DE_RECOMBINATION = 0.7                                    #recombination parameter for differential evolution minimization
    POLISHING_X_TOL = 1e-4                                    #parameter tolerance for polishing minimization
    POLISHING_G_TOL = 1e-5                                    #gradient tolerance for polishing minimization
    FIT_PROGRESS_FIG_SIZE = (3*6.4,2*4.6)                     #(width, height) of the fit progress figure
    PP_RAVG_POINTS = 10                                       #how many points to average over for the polishing minimization progress plots
    OVERLAP_COMPARISON_DIR_NAME = 'overlap_comparison_images' #name of directory holding overlap comparison images
    OCTET_OVERLAP_COMPARISON_FIGURE_WIDTH = 3*6.4             #width of the octet overlap comparison figures
    FIT_RESULT_TEXT_FILE_NAME = 'warping_parameters.txt'      #the name of the fit result text file that gets written out

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
        #same with the best fit warp
        self._best_fit_warp = None

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

    def doFit(self,fixed=None,normalize=None,float_p1p2_in_polish_fit=False,max_radial_warp=10.,max_tangential_warp=10.,
             p1p2_polish_lasso_lambda=0.,polish=True,print_every=1,maxiter=1000) :
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
        self.fitpars.setFirstMinimizationResults(de_result)
        #do the polishing minimization
        if polish :
            result = self.__runPolishMinimization(float_p1p2_in_polish_fit,p1p2_polish_lasso_lambda,maxiter)
        else :
            result=de_result
        polish_minimization_done_time = time.time()
        #set the final fit parameter values
        self.fitpars.setFinalResults(result)
        #record the minimization run times
        self.init_min_runtime = init_minimization_done_time-minimization_start_time
        self.polish_min_runtime = polish_minimization_done_time-init_minimization_done_time
        #run all the post-processing stuff
        self.__runPostProcessing(de_result.nfev)

    def checkFit(self,fixed=None,normalize=None,float_p1p2_in_polish_fit=False,max_radial_warp=10.,max_tangential_warp=10.,
                 p1p2_polish_lasso_lambda=0.,polish=True) :
        """
        A function to print some information about how the fits will proceed with the current settings
        (see "doFit" function above for what the arguments to this function are)
        """
        #make the set of fit parameters
        self.fitpars = FitParameterSet(fixed,normalize,max_radial_warp,max_tangential_warp,self.warpset.warp)
        #stuff from the differential evolution minimization setup
        warp_logger.info('For global minimization setup:')
        _, _, _ = self.__getGlobalSetup()
        #stuff for the polishing minimization, if applicable
        if polish :
            warp_logger.info('For polishing minimization setup:')
            _, _, _, _ = self.__getPolishingSetup(float_p1p2_in_polish_fit,p1p2_polish_lasso_lambda)
            if p1p2_polish_lasso_lambda!=0 :
                warp_logger.info(f'p1 and p2 will be LASSOed in the polishing minimization with lambda = {p1p2_polish_lasso_lambda}')
            else :
                warp_logger.info('p1 and p2 will not be LASSOed in the polishing minimization')
        else :
            warp_logger.info('Polishing minimization will be skipped')

    #################### MINIMIZATION FUNCTIONS ####################

    # !!!!!! For the time being, these functions don't correctly describe dependence on k4, k5, or k6 !!!!!!

    #The function whose return value is minimized by the fitting
    def _evalCamWarpOnAlignmentSet(self,pars) :
        self.minfunc_calls+=1
        #first fix the parameter list so the warp functions always see vectors of the same length
        warp_pars = self.fitpars.warpParsFromFitPars(pars)
        #update the warp with the new parameters
        self.warpset.updateCameraParams(warp_pars)
        #then warp the images
        self.warpset.warpLoadedImages(skip_corners=self.skip_corners)
        #reload the (newly-warped) images into the alignment set
        self.alignset.updateRectangleImages([warpimg for warpimg in self.warpset.images if not (self.skip_corners and warpimg.is_corner_only)])
        #check the warp amounts to see if the sample should be realigned
        rad_warp = self.warpset.warp.maxRadialDistortAmount(warp_pars)
        tan_warp = self.warpset.warp.maxTangentialDistortAmount(warp_pars)
        align_strategy = 'overwrite' if (abs(rad_warp)<self.fitpars.max_rad_warp) and (tan_warp<self.fitpars.max_tan_warp) else 'shift_only'
        #align the images and get the cost
        aligncost = self.alignset.align(skip_corners=self.skip_corners,
                                        write_result=False,
                                        return_on_invalid_result=True,
                                        alreadyalignedstrategy=align_strategy,
                                        warpwarnings=True,
                                    )
        #compute the cost from the LASSO constraint on the specified parameters
        lasso_cost=self.fitpars.getLassoCost(pars)
        #sum the costs
        cost = aligncost+lasso_cost
        #add to the lists to plot
        self.costs.append(cost if cost<1e10 else -0.1)
        self.max_radial_warps.append(rad_warp)
        self.max_tangential_warps.append(tan_warp)
        #print progress if requested
        if self.minfunc_calls%self.print_every==0 :
            warp_logger.info(self.warpset.warp.paramString())
            msg = f'  Call {self.minfunc_calls} cost={cost:.04f}(={aligncost:.04f}+{lasso_cost:.04f})' 
            msg+=f' (radial warp={self.max_radial_warps[-1]:.02f}, tangential warp={self.max_tangential_warps[-1]:.02f})'
            warp_logger.info(msg)
        #return the cost from the alignment
        return cost

    #call the warp's max radial distort amount function with the corrected parameters
    def _maxRadialDistortAmountForConstraint(self,pars) :
        return self.warpset.warp.maxRadialDistortAmount(self.fitpars.warpParsFromFitPars(pars))

    #get the correctly-formatted Jacobian of the maximum radial distortion amount constraint 
    def _maxRadialDistortAmountJacobianForConstraint(self,pars) :
        warpresult = np.array(self.warpset.warp.maxRadialDistortAmountJacobian(self.fitpars.warpParsFromFitPars(pars)))
        return self.fitpars.fitParsFromWarpPars(warpresult)

    #call the warp's max tangential distort amount function with the corrected parameters
    def _maxTangentialDistortAmountForConstraint(self,pars) :
        return self.warpset.warp.maxTangentialDistortAmount(self.fitpars.warpParsFromFitPars(pars))

    #get the correctly-formatted Jacobian of the maximum tangential distortion amount constraint 
    def _maxTangentialDistortAmountJacobianForConstraint(self,pars) :
        warpresult = np.array(self.warpset.warp.maxTangentialDistortAmountJacobian(self.fitpars.warpParsFromFitPars(pars)))
        return self.fitpars.fitParsFromWarpPars(warpresult)

    #function to run global minimization with differential evolution and return the result
    def __runDifferentialEvolution(self,maxiter) :
        #set up the parameter bounds and constraints and get the initial population
        parameter_bounds, constraints, initial_population = self.__getGlobalSetup()
        self._de_population_size = len(initial_population)
        #run the minimization
        warp_logger.info('Starting initial minimization....')
        self.skip_corners = True
        with cd(self.working_dir) :
            try :
                result=scipy.optimize.differential_evolution(
                    func=self._evalCamWarpOnAlignmentSet,
                    bounds=parameter_bounds,
                    strategy='best2bin',
                    maxiter=maxiter,
                    tol=self.DE_TOLERANCE,
                    mutation=self.DE_MUTATION,
                    recombination=self.DE_RECOMBINATION,
                    polish=False, #we'll do our own polishing minimization
                    init=initial_population,
                    constraints=constraints
                    )
            except Exception :
                raise WarpingError('Something failed in the initial minimization!')
        warp_logger.info(f'Initial minimization completed {"successfully" if result.success else "UNSUCCESSFULLY"} in {result.nfev} evaluations.')
        return result

    #function to run local polishing minimiation with trust-constr and return the result
    def __runPolishMinimization(self,float_p1p2_in_polish_fit,p1p2_polish_lasso_lambda,maxiter) :
        #set up the parameter bounds, constraints, initial values, and relative step sizes
        parameter_bounds, constraints, init_pars, rel_steps = self.__getPolishingSetup(float_p1p2_in_polish_fit,p1p2_polish_lasso_lambda)
        #call minimize with trust_constr
        warp_logger.info('Starting polishing minimization....')
        with cd(self.working_dir) :
            try :
                result=scipy.optimize.minimize(
                    fun=self._evalCamWarpOnAlignmentSet,
                    x0=init_pars,
                    method='trust-constr',
                    bounds=parameter_bounds,
                    constraints=constraints,
                    options={'xtol':self.POLISHING_X_TOL,'gtol':self.POLISHING_G_TOL,'finite_diff_rel_step':rel_steps,'maxiter':maxiter}
                    )
            except Exception :
                raise WarpingError('Something failed in the polishing minimization!')
        msg = f'Final minimization completed {"successfully" if result.success else "UNSUCCESSFULLY"} in {result.nfev} evaluations '
        term_conds = {0:'max iterations',1:'gradient tolerance',2:'parameter tolerance',3:'callback function'}
        msg+=f'due to compliance with {term_conds[result.status]} criteria.'
        warp_logger.info(msg)
        return result

    #################### VISUALIZATION/OUTPUT FUNCTIONS ####################

    #helper function to run all of the various post-processing functions after the fit is done
    def __runPostProcessing(self,n_initial_fevs) :
        #make the fit progress plots
        self.init_its, self.polish_its = self.__makeFitProgressPlots(n_initial_fevs)
        #use the fit result to make the best fit warp object
        self.warpset.updateCameraParams(self.fitpars.best_fit_warp_parameters)
        self._best_fit_warp = copy.deepcopy(self.warpset.warp)
        warp_logger.info('Best fit parameters:')
        self._best_fit_warp.printParams()
        #save the figure of the best-fit warp fields
        with cd(self.working_dir) :
            self._best_fit_warp.makeWarpAmountFigure()
        #write out the set of alignment comparison images
        self.raw_cost, self.best_cost = self.__makeBestFitAlignmentComparisonImages()
        #write out the fit result text file
        self.__writeFitResultTextFile()

    #function to plot the costs and warps over all the iterations of the fit
    def __makeFitProgressPlots(self,ninitev) :
        nfev = len(self.costs)-ninitev
        inititers  = np.linspace(0,ninitev,ninitev,endpoint=False)
        finaliters = np.linspace(ninitev,len(self.costs),nfev,endpoint=False)
        f,ax = plt.subplots(2,3,figsize=self.FIT_PROGRESS_FIG_SIZE)
        #global minimization costs
        ax[0][0].plot(inititers,self.costs[:ninitev],label='all')
        ngenerations = int(ninitev/self._de_population_size)
        pop_avg_costs = []
        for ig in range(ngenerations) :
            pop_avg_cost = np.mean(np.array(self.costs[ig*self._de_population_size:(ig+1)*self._de_population_size]))
            for ip in range(self._de_population_size) :
                pop_avg_costs.append(pop_avg_cost)
        ax[0][0].plot(inititers[:len(pop_avg_costs)],pop_avg_costs,label='population averages')
        ax[0][0].set_xlabel('initial minimization iteration')
        ax[0][0].set_ylabel('cost')
        ax[0][0].legend(loc='best')
        #global minimization radial warps
        ax[0][1].plot(inititers,self.max_radial_warps[:ninitev],label='all')
        pop_avg_rad_warps = []
        for ig in range(ngenerations) :
            pop_avg_rad_warp = np.mean(np.array(self.max_radial_warps[ig*self._de_population_size:(ig+1)*self._de_population_size]))
            for ip in range(self._de_population_size) :
                pop_avg_rad_warps.append(pop_avg_rad_warp)
        ax[0][1].plot(inititers[:len(pop_avg_rad_warps)],pop_avg_rad_warps,label='population averages')
        ax[0][1].set_xlabel('initial minimization iteration')
        ax[0][1].set_ylabel('max radial warp')
        ax[0][1].legend(loc='best')
        #global minimization tangential warps
        ax[0][2].plot(inititers,self.max_tangential_warps[:ninitev],label='all')
        pop_avg_tan_warps = []
        for ig in range(ngenerations) :
            pop_avg_tan_warp = np.mean(np.array(self.max_tangential_warps[ig*self._de_population_size:(ig+1)*self._de_population_size]))
            for ip in range(self._de_population_size) :
                pop_avg_tan_warps.append(pop_avg_tan_warp)
        ax[0][2].plot(inititers[:len(pop_avg_tan_warps)],pop_avg_tan_warps,label='population averages')
        ax[0][2].set_xlabel('initial minimization iteration')
        ax[0][2].set_ylabel('max tangential warp')
        ax[0][2].legend(loc='best')
        #polishing minimization costs
        ax[1][0].plot(finaliters,self.costs[ninitev:],label='all')
        running_avgs = [np.mean(np.array(self.costs[ninitev+i:ninitev+i+self.PP_RAVG_POINTS])) for i in range(nfev-self.PP_RAVG_POINTS)]
        ax[1][0].plot(finaliters[self.PP_RAVG_POINTS:],running_avgs,label=f'{self.PP_RAVG_POINTS}-point running average')
        ax[1][0].set_xlabel('final minimization iteration')
        ax[1][0].set_ylabel('cost')
        ax[1][0].legend(loc='best')
        #polishing minimization radial warps
        ax[1][1].plot(finaliters,self.max_radial_warps[ninitev:],label='all')
        running_avgs = [np.mean(np.array(self.max_radial_warps[ninitev+i:ninitev+i+self.PP_RAVG_POINTS])) for i in range(nfev-self.PP_RAVG_POINTS)]
        ax[1][1].plot(finaliters[self.PP_RAVG_POINTS:],running_avgs,label=f'{self.PP_RAVG_POINTS}-point running average')
        ax[1][1].set_xlabel('final minimization iteration')
        ax[1][1].set_ylabel('max radial warp')
        ax[1][1].legend(loc='best')
        #polishing minimization tangential warps
        ax[1][2].plot(finaliters,self.max_tangential_warps[ninitev:],label='all')
        running_avgs = [np.mean(np.array(self.max_tangential_warps[ninitev+i:ninitev+i+self.PP_RAVG_POINTS])) for i in range(nfev-self.PP_RAVG_POINTS)]
        ax[1][2].plot(finaliters[self.PP_RAVG_POINTS:],running_avgs,label=f'{self.PP_RAVG_POINTS}-point running average')
        ax[1][2].set_xlabel('final minimization iteration')
        ax[1][2].set_ylabel('max tangential warp')
        ax[1][2].legend(loc='best')
        with cd(self.working_dir) :
            plt.savefig('fit_progress.png')
            plt.close()
        return ninitev, nfev

    #function to save alignment comparison visualizations in a new directory inside the working directory
    def __makeBestFitAlignmentComparisonImages(self) :
        warp_logger.info('writing out warping/alignment comparison images')
        #make sure the best fit warp exists (which means the warpset is updated with the best fit parameters)
        if self._best_fit_warp is None :
            raise WarpingError('Do not call __makeBestFitAlignmentComparisonImages until after the best fit warp has been set!')
        #make sure the plot directory exists
        with cd(self.working_dir) :
            if not os.path.isdir(self.OVERLAP_COMPARISON_DIR_NAME) :
                os.mkdir(self.OVERLAP_COMPARISON_DIR_NAME)
        #build octets and singlets from the alignment set's overlaps
        all_olaps = self.alignset.overlaps
        olap_octet_p1s   = [olap1.p1 for olap1 in all_olaps if len([olap2 for olap2 in all_olaps if olap2.p1==olap1.p1])==8]
        olap_singlet_p1s = [olap1.p1 for olap1 in all_olaps if len([olap2 for olap2 in all_olaps if olap2.p1==olap1.p1])!=8 and olap1.p2 not in olap_octet_p1s]
        nclip1 = all_olaps[0].nclip
        nclip2 = nclip1+2
        #start by aligning the raw, unwarped images and getting their shift comparison information/images and the raw p1 images
        self.alignset.updateRectangleImages(self.warpset.images,usewarpedimages=False)
        rawcost = self.alignset.align(write_result=False,alreadyalignedstrategy="overwrite",warpwarnings=True)
        raw_olap_comps = self.alignset.getOverlapComparisonImagesDict()
        raw_octet_p1_images = [copy.deepcopy([r.image for r in self.alignset.rectangles if r.n==octetp1][0]) for octetp1 in olap_octet_p1s]
        #next warp and align the images with the best fit warp and do the same thing
        self.warpset.warpLoadedImages()
        self.alignset.updateRectangleImages(self.warpset.images)
        bestcost = self.alignset.align(write_result=False,alreadyalignedstrategy="overwrite",warpwarnings=True)
        warped_olap_comps = self.alignset.getOverlapComparisonImagesDict()
        warped_octet_p1_images = [copy.deepcopy([r.image for r in self.alignset.rectangles if r.n==octetp1][0]) for octetp1 in olap_octet_p1s]
        #print the cost differences
        warp_logger.info(f'Alignment cost from raw images = {rawcost:.08f}; alignment cost from warped images = {bestcost:.08f} ({(100*(1.-bestcost/rawcost)):.04f}% reduction)')
        #write out the octet comparison figures
        #for each octet
        for octetp1,raw_p1_image,warped_p1_image in zip(olap_octet_p1s,raw_octet_p1_images,warped_octet_p1_images) :
            #get the ordered lists of overlap comparison images
            rois = []; raois = []; wois = []; waois = []
            key_order = [1,2,3,4,6,7,8,9]
            for c in key_order :
                rois.append([raw_olap_comps[oli][0] for oli in raw_olap_comps.keys() if oli[1]==octetp1 and oli[0]==c][0])
                raois.append([raw_olap_comps[oli][1] for oli in raw_olap_comps.keys() if oli[1]==octetp1 and oli[0]==c][0])
                wois.append([warped_olap_comps[oli][0] for oli in warped_olap_comps.keys() if oli[1]==octetp1 and oli[0]==c][0])
                waois.append([warped_olap_comps[oli][1] for oli in warped_olap_comps.keys() if oli[1]==octetp1 and oli[0]==c][0])
            #make the total images
            total_image_shape = (nclip1+rois[3].shape[0]+nclip1,nclip1+rois[1].shape[1]+nclip1,rois[0].shape[2]); total_image_dtype = rois[0].dtype
            total_roi  = np.zeros(total_image_shape,dtype=total_image_dtype)
            total_raoi = np.zeros(total_image_shape,dtype=total_image_dtype) 
            total_woi  = np.zeros(total_image_shape,dtype=total_image_dtype)
            total_waoi = np.zeros(total_image_shape,dtype=total_image_dtype)
            #fill the top left corners
            y2=rois[7].shape[0]  
            x2=rois[7].shape[1]
            total_roi[nclip1:y2+nclip1,nclip1:x2+nclip1,:]  = (1./3.)*(rois[7][:,:,:]+rois[6][:,:x2,:]+rois[4][:y2,:,:])
            total_raoi[nclip1+nclip2:y2+nclip1-nclip2,nclip1+nclip2:x2+nclip1-nclip2,:] = (1./3.)*(raois[7][:,:,:]+raois[6][:,:x2-2*nclip2,:]+raois[4][:y2-2*nclip2,:,:])
            total_woi[nclip1:y2+nclip1,nclip1:x2+nclip1,:]  = (1./3.)*(wois[7][:,:,:]+wois[6][:,:x2,:]+wois[4][:y2,:,:])
            total_waoi[nclip1+nclip2:y2+nclip1-nclip2,nclip1+nclip2:x2+nclip1-nclip2,:] = (1./3.)*(waois[7][:,:,:]+waois[6][:,:x2-2*nclip2,:]+waois[4][:y2-2*nclip2,:,:])
            #fill the center tops
            x1=x2; x2=rois[6].shape[1]-rois[5].shape[1]
            total_roi[nclip1:y2+nclip1,nclip1+x1:x2+nclip1,:]  = rois[6][:,x1:x2,:]
            total_raoi[nclip1+nclip2:y2+nclip1-nclip2,nclip1+x1:x2+nclip1,:] = raois[6][:,x1:x2,:]
            total_woi[nclip1:y2+nclip1,nclip1+x1:x2+nclip1,:]  = wois[6][:,x1:x2,:]
            total_waoi[nclip1+nclip2:y2+nclip1-nclip2,nclip1+x1:x2+nclip1,:] = waois[6][:,x1:x2,:]
            #fill the top right corners
            x1=x2
            total_roi[nclip1:y2+nclip1,nclip1+x1:-nclip1,:]  = (1./3.)*(rois[5][:,:,:]+rois[6][:,x1:,:]+rois[3][:y2,:,:])
            total_raoi[nclip1+nclip2:y2+nclip1-nclip2,nclip1+x1+nclip2:-(nclip1+nclip2),:] = (1./3.)*(raois[5][:,:,:]+raois[6][:,x1:,:]+raois[3][:y2-2*nclip2,:,:])
            total_woi[nclip1:y2+nclip1,nclip1+x1:-nclip1,:]  = (1./3.)*(wois[5][:,:,:]+wois[6][:,x1:,:]+wois[3][:y2,:,:])
            total_waoi[nclip1+nclip2:y2+nclip1-nclip2,nclip1+x1+nclip2:-(nclip1+nclip2),:] = (1./3.)*(waois[5][:,:,:]+waois[6][:,x1:,:]+waois[3][:y2-2*nclip2,:,:])
            #fill the center lefts
            y1=rois[7].shape[0]; y2=rois[4].shape[0]-rois[2].shape[0]
            x2=rois[7].shape[1]
            total_roi[nclip1+y1:y2+nclip1,nclip1:x2+nclip1,:]  = rois[4][y1:y2,:,:]
            total_raoi[nclip1+y1:y2+nclip1,nclip1+nclip2:x2+nclip1-nclip2,:] = raois[4][y1:y2,:,:]
            total_woi[nclip1+y1:y2+nclip1,nclip1:x2+nclip1,:]  = wois[4][y1:y2,:,:]
            total_waoi[nclip1+y1:y2+nclip1,nclip1+nclip2:x2+nclip1-nclip2,:] = waois[4][y1:y2,:,:]
            #fill the center rights
            x1=rois[6].shape[1]-rois[5].shape[1]
            total_roi[nclip1+y1:y2+nclip1,nclip1+x1:-nclip1,:]  = rois[3][y1:y2,:,:]
            total_raoi[nclip1+y1:y2+nclip1,nclip1+x1+nclip2:-(nclip1+nclip2),:] = raois[3][y1:y2,:,:]
            total_woi[nclip1+y1:y2+nclip1,nclip1+x1:-nclip1,:]  = wois[3][y1:y2,:,:]
            total_waoi[nclip1+y1:y2+nclip1,nclip1+x1+nclip2:-(nclip1+nclip2),:] = waois[3][y1:y2,:,:]
            #fill the bottom left corners
            y1=rois[4].shape[0]-rois[2].shape[0]; 
            x2=rois[2].shape[1]
            total_roi[nclip1+y1:-nclip1,nclip1:x2+nclip1,:]  = (1./3.)*(rois[2][:,:,:]+rois[1][:,:x2,:]+rois[4][y1:,:,:])
            total_raoi[nclip1+y1+nclip2:-(nclip1+nclip2),nclip1+nclip2:x2+nclip1-nclip2,:] = (1./3.)*(raois[2][:,:,:]+raois[1][:,:x2-2*nclip2,:]+raois[4][y1:,:,:])
            total_woi[nclip1+y1:-nclip1,nclip1:x2+nclip1,:]  = (1./3.)*(wois[2][:,:,:]+wois[1][:,:x2,:]+wois[4][y1:,:,:])
            total_waoi[nclip1+y1+nclip2:-(nclip1+nclip2),nclip1+nclip2:x2+nclip1-nclip2,:] = (1./3.)*(waois[2][:,:,:]+waois[1][:,:x2-2*nclip2,:]+waois[4][y1:,:,:])
            #fill the center bottoms
            x1=x2; x2=rois[1].shape[1]-rois[0].shape[1]
            total_roi[nclip1+y1:-nclip1,nclip1+x1:x2+nclip1,:]  = rois[1][:,x1:x2,:]
            total_raoi[nclip1+y1+nclip2:-(nclip1+nclip2),nclip1+x1:x2+nclip1,:] = raois[1][:,x1:x2,:]
            total_woi[nclip1+y1:-nclip1,nclip1+x1:x2+nclip1,:]  = wois[1][:,x1:x2,:]
            total_waoi[nclip1+y1+nclip2:-(nclip1+nclip2),nclip1+x1:x2+nclip1,:] = waois[1][:,x1:x2,:]
            #fill the bottom right corners
            x1=x2
            total_roi[nclip1+y1:-nclip1,nclip1+x1:-nclip1,:]  = (1./3.)*(rois[0][:,:,:]+rois[1][:,x1:,:]+rois[3][y1:,:,:])
            total_raoi[nclip1+y1+nclip2:-(nclip1+nclip2),nclip1+x1+nclip2:-(nclip1+nclip2),:] =(1./3.)*(raois[0][:,:,:]+raois[1][:,x1:,:]+raois[3][y1:,:,:])
            total_woi[nclip1+y1:-nclip1,nclip1+x1:-nclip1,:]  = (1./3.)*(wois[0][:,:,:]+wois[1][:,x1:,:]+wois[3][y1:,:,:])
            total_waoi[nclip1+y1+nclip2:-(nclip1+nclip2),nclip1+x1+nclip2:-(nclip1+nclip2),:] = (1./3.)*(waois[0][:,:,:]+waois[1][:,x1:,:]+waois[3][y1:,:,:])
            #fill in the rest of the images' clipped portions in just magenta
            total_roi[:,:,0] = np.where(total_roi[:,:,0]==0,raw_p1_image[:,:]/1000.,total_roi[:,:,0])
            total_roi[:,:,2] = np.where(total_roi[:,:,2]==0,raw_p1_image[:,:]/2000.,total_roi[:,:,2])
            total_raoi[:,:,0] = np.where(total_raoi[:,:,0]==0,raw_p1_image[:,:]/1000.,total_raoi[:,:,0])
            total_raoi[:,:,2] = np.where(total_raoi[:,:,2]==0,raw_p1_image[:,:]/2000.,total_raoi[:,:,2])
            total_woi[:,:,0] = np.where(total_woi[:,:,0]==0,warped_p1_image[:,:]/1000.,total_woi[:,:,0])
            total_woi[:,:,2] = np.where(total_woi[:,:,2]==0,warped_p1_image[:,:]/2000.,total_woi[:,:,2])
            total_waoi[:,:,0] = np.where(total_waoi[:,:,0]==0,warped_p1_image[:,:]/1000.,total_waoi[:,:,0])
            total_waoi[:,:,2] = np.where(total_waoi[:,:,2]==0,warped_p1_image[:,:]/2000.,total_waoi[:,:,2])
            #plot and save the entire images
            total_images = [total_roi,total_raoi,total_woi,total_waoi]
            image_names = [f'octet_p1={octetp1}_raw_overlap_comparisons',
                           f'octet_p1={octetp1}_raw_aligned_overlap_comparisons',
                           f'octet_p1={octetp1}_warped_overlap_comparisons',
                           f'octet_p1={octetp1}_warped_aligned_overlap_comparisons']
            for total_image,image_name in zip(total_images,image_names) :
                f,ax = plt.subplots(figsize=(self.OCTET_OVERLAP_COMPARISON_FIGURE_WIDTH,
                                             int((total_image.shape[0]/total_image.shape[1])*self.OCTET_OVERLAP_COMPARISON_FIGURE_WIDTH)))
                ax.imshow(total_image)
                ax.set_title(image_name.replace('_',' '))
                with cd(os.path.join(self.working_dir,self.OVERLAP_COMPARISON_DIR_NAME)) :
                    plt.savefig(f'{image_name}.png')
                    plt.close()
        #plot the singlet overlap comparisons
        for overlap_identifier in raw_olap_comps.keys() :
            code = overlap_identifier[0]
            p1 = overlap_identifier[1]
            if p1 not in olap_singlet_p1s :
                continue
            p2 = overlap_identifier[2]
            fn   = overlap_identifier[3]
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
            order[0].imshow(raw_olap_comps[overlap_identifier][0])
            order[0].set_title('raw overlap images')
            order[1].imshow(raw_olap_comps[overlap_identifier][1])
            order[1].set_title('raw overlap images aligned')
            order[2].imshow(warped_olap_comps[overlap_identifier][0])
            order[2].set_title('warped overlap images')
            order[3].imshow(warped_olap_comps[overlap_identifier][1])
            order[3].set_title('warped overlap images aligned')
            with cd(os.path.join(self.working_dir,self.OVERLAP_COMPARISON_DIR_NAME)) :
                plt.savefig(fn)
                plt.close()
        #return the pre- and post-fit alignment costs
        return rawcost, bestcost

    #helper function to write the parameter text file
    def __writeFitResultTextFile(self) :
        to_write = {'n':str(self.n),
                    'm':str(self.m),
                   }
        for k,v in self.fitpars.result_text_file_lines.items() :
            to_write[k] = v
        max_r_x, max_r_y = self.warpset.warp._getMaxDistanceCoords()
        to_write['max_r_x_coord']             = str(max_r_x)
        to_write['max_r_y_coord']             = str(max_r_y)
        to_write['max_r']                     = str(math.sqrt((max_r_x)**2+(max_r_y)**2))
        to_write['max_radial_warp']           = str(self.warpset.warp.maxRadialDistortAmount(self.fitpars.best_fit_warp_parameters))
        to_write['max_tangential_warp']       = str(self.warpset.warp.maxTangentialDistortAmount(self.fitpars.best_fit_warp_parameters))
        to_write['initial_fit_iterations']    = str(self.init_its)
        to_write['polishing_fit_iterations']  = str(self.polish_its)
        to_write['initial_minimization_time'] = str(self.init_min_runtime)
        to_write['polish_minimization_time']  = str(self.polish_min_runtime)
        to_write['raw_cost']                  = str(self.raw_cost)
        to_write['best_cost']                 = str(self.best_cost)
        to_write['cost_reduction']            = f'{(100*(1.-self.best_cost/self.raw_cost)):.2f}%'
        max_key_width = 0; max_value_width = 0
        for k,v in to_write.items() :
            if len(k)>max_key_width :
                max_key_width=len(k)
            if len(v)>max_value_width :
                max_value_width=len(v)
        with cd(self.working_dir) :
            with open(self.FIT_RESULT_TEXT_FILE_NAME,'w') as fp :
                for k,v in to_write.items() :
                    fp.write(f'{k:<{max_key_width+3}}{v:<{max_value_width}}\n')

    #################### OTHER PRIVATE HELPER FUNCTIONS ####################

    # helper function to create and return a new alignmentSet object that's set up to run on the identified set of images/overlaps
    def __initializeAlignmentSet(self, *, overlaps) :
        #If this is running on my Mac I want to be asked which GPU device to use because it doesn't default to the AMD compute unit....
        customGPUdevice = True if platform.system()=='Darwin' else False
        a = AlignmentSet(self.dbload_top_dir,self.working_dir,self.samp_name,interactive=customGPUdevice,useGPU=True,
                         selectoverlaps=rectangleoroverlapfilter(overlaps, compatibility=True),onlyrectanglesinoverlaps=True)
        return a

    #helper function to return the parameter bounds, constraints, and initial population for the global minimization
    def __getGlobalSetup(self) :
        bounds = self.fitpars.getFitParameterBounds()
        constraints = self.__getConstraints()
        initial_population = self.fitpars.getInitialPopulation()
        return bounds, constraints, initial_population

    #helper function to return the lists of parameter bounds, constraints, initial values, and relative step sizes for the polishing fit 
    def __getPolishingSetup(self,float_p1p2_in_polish_fit,p1p2_polish_lasso_lambda) :
        bounds, initial_values, relative_step_sizes = self.fitpars.getPolishingSetup(float_p1p2_in_polish_fit,p1p2_polish_lasso_lambda)
        constraints = self.__getConstraints()
        return bounds, constraints, initial_values, relative_step_sizes

    #helper function to make the list of constraints
    def __getConstraints(self) :
        constraints = []; names_to_print = []
        #if the radial warp is being fit, and the max_radial_warp is defined, add the max_radial_warp constraint
        if self.fitpars.radial_warp_floating and self.fitpars.max_rad_warp>0 :
            constraints.append(scipy.optimize.NonlinearConstraint(
                self._maxRadialDistortAmountForConstraint,
                -1.*self.fitpars.max_rad_warp,
                self.fitpars.max_rad_warp,
                jac=self._maxRadialDistortAmountJacobianForConstraint,
                hess=scipy.optimize.BFGS()
                )
            )
            names_to_print.append(f'max radial warp={self.fitpars.max_rad_warp} pixels')
        #if the tangential warping is being fit, and the max_tangential_warp is defined, add the max_tangential_warp constraint
        if self.fitpars.tangential_warp_floating and self.fitpars.max_tan_warp>0 :
            constraints.append(scipy.optimize.NonlinearConstraint(
                self._maxTangentialDistortAmountForConstraint,
                0.,
                self.fitpars.max_tan_warp,
                jac=self._maxTangentialDistortAmountJacobianForConstraint,
                hess=scipy.optimize.BFGS()
                )
            )
            names_to_print.append(f'max tangential warp={self.fitpars.max_tan_warp} pixels')
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
        if len(constraints)==1 : #if there's only one it doesn't get passed as a list (thanks scipy)
            return constraints[0]
        else :
            return constraints
