#imports
import time, scipy, logging
from ...shared.samplemetadata import MetadataSummary
from .config import CONST
from .utilities import WarpFitResult, FieldLog
from .warp import CameraWarp
from .fitparameterset import FitParameterSet

class WarpFit :
    """
    A class for performing a single warp fit in a specific way
    """

    #################### CLASS CONSTANTS ####################

    DE_TOLERANCE = 0.03     #tolerance for the differential evolution minimization
    DE_MUTATION = (0.2,0.8) #mutation bounds for differential evolution minimization
    DE_RECOMBINATION = 0.7  #recombination parameter for differential evolution minimization

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,warpsample,octet,logger) :
        """
        warpsample = the WarpingSample object for the given octet (used to get images)
        octet = the OverlapOctet object specifying the relevant rectangles/overlaps
        logger = the logger object to use (likely the globallogger of the warpingcohort)
        """
        #set the warping sample that will be used to align the overlaps
        self.__warpsample = warpsample
        #keep the octet and logger around also
        self.__octet = octet
        self.logger = logger
        #get the overlap objects and unwarped images to hold in memory
        self.__overlaps = []
        olap_ns = [octet.olap_1_n,octet.olap_2_n,octet.olap_3_n,octet.olap_4_n,
                   octet.olap_6_n,octet.olap_7_n,octet.olap_8_n,octet.olap_9_n]
        p2_rect_ns = []
        for olap_n in olap_ns :
            for olap in self.__warpsample.overlaps :
                if olap.n==olap_n :
                    self.__overlaps.append(olap)
                    p2_rect_ns.append(olap.p2)
                    break
        self.__unwarped_images_by_rect_i = {}
        for rect_n in [octet.p1_rect_n,*(p2_rect_ns)] :
            for ri,rect in enumerate(self.__warpsample.rectangles) :
                if rect.n==rect_n :
                    self.__unwarped_images_by_rect_i[ri] = rect.image[:,:,0]

    def run(self,max_iters,fixed,init_pars,init_bounds,max_rad_warp,max_tan_warp) :
        """
        Actually run the fit and return its WarpFitResult, FieldLog, and MetadataSummary objects
        
        max_iters = the maximum number of differential evolution iterations to run
        fixed = a list of parameter names that should be kept fixed during fitting
        init_pars = a dictionary of initial parameter values
        init_bounds = a dictionary of initial parameter bounds
        max_rad_warp = the maximum amount of radial warping in pixels to allow (used for a constraint)
        max_tan_warp = the maximum amount of tangential warping in pixels to allow (used for a constraint)
        """
        #initialize the warp object that will be used
        warpkwargs = {}
        ex_im = list(self.__unwarped_images_by_rect_i.values())[0]
        warpkwargs['n'] = ex_im.shape[1]
        warpkwargs['m'] = ex_im.shape[0]
        self.__warp = CameraWarp(**warpkwargs)
        #initialize the fit parameter set that will be used
        self.__fitpars = FitParameterSet(fixed,init_pars,init_bounds,max_rad_warp,max_tan_warp,self.__warp,self.logger)
        #make the iteration counter
        self.minfunc_calls=0
        #silence the WarpingSample logger so we don't get a ton of output each time the images are aligned
        self.__warpsample.logger.setLevel(logging.WARN)
        #do the global minimization
        minimization_start_time = time.time()
        de_result = self.__run_differential_evolution(max_iters)
        minimization_done_time = time.time()
        #set the fit parameter values after the minimization
        self.__fitpars.set_final_results(de_result)
        #Create the WarpFitResult
        self.__warpsample.update_rectangle_images(self.__unwarped_images_by_rect_i,self.__octet.p1_rect_n)
        raw_cost = self.__get_fit_cost()
        self.__warp.updateParams(self.__fitpars.best_fit_warp_parameters)
        self.__warpsample.update_rectangle_images(self.__get_warped_images_by_rect_i(),self.__octet.p1_rect_n)
        best_cost = self.__get_fit_cost()
        wfr = WarpFitResult(self.__warpsample.SlideID,self.__octet.p1_rect_n,
                            self.__warp.n,self.__warp.m,
                            *(self.__fitpars.best_fit_warp_parameters),
                            self.__warp.maxRadialDistortAmount(self.__fitpars.best_fit_warp_parameters),
                            self.__warp.maxTangentialDistortAmount(self.__fitpars.best_fit_warp_parameters),
                            self.minfunc_calls,
                            minimization_done_time-minimization_start_time,
                            raw_cost,best_cost,(raw_cost-best_cost)/raw_cost
                           )
        #Create the FieldLog and MetadataSummary objects
        rects = [self.__warpsample.rectangles[i] for i in self.__unwarped_images_by_rect_i.keys()]
        fls = [FieldLog(r.file,r.n) for r in rects]
        mds = MetadataSummary(self.__warpsample.SlideID,self.__warpsample.Project,
                              self.__warpsample.Cohort,self.__warpsample.microscopename,
                              str(min([r.t for r in rects])),str(max([r.t for r in rects])))
        #Return everything
        return wfr,fls,mds

    #################### FUNCTIONS CALLED DURING MINIMIZATION ####################

    def _minimization_function(self,pars) :
        """
        The function that is repeatedly called whose return value is minimized by running differential evolution
        """
        self.minfunc_calls+=1
        #first fix the parameter list so the warp functions always see vectors of the same length
        warp_pars = self.__fitpars.warp_pars_from_fit_pars(pars)
        #update the warp with the new parameters
        self.__warp.updateParams(warp_pars)
        #then warp the images and replace them in the rectangle/overlap objects
        rectangle_images_for_update = self.__get_warped_images_by_rect_i()
        self.__warpsample.update_rectangle_images(rectangle_images_for_update,self.__octet.p1_rect_n)
        #align the images and get the cost
        cost = self.__get_fit_cost()
        #print progress if applicable
        if self.minfunc_calls%CONST.PRINT_EVERY==0 :
            msg = f'Call {self.minfunc_calls} cost={cost:.05f} ({self.__warp.paramString()})'
            self.logger.debug(msg)
        #return the cost from the alignment
        return cost

    def _maxRadialDistortAmountForConstraint(self,pars) :
        """
        call the warp's max radial distort amount function with the corrected parameters
        """
        return self.__warp.maxRadialDistortAmount(self.__fitpars.warp_pars_from_fit_pars(pars))

    def _maxRadialDistortAmountJacobianForConstraint(self,pars) :
        """
        get the correctly-formatted Jacobian of the maximum radial distortion amount constraint 
        """
        wr = np.array(self.__warp.maxRadialDistortAmountJacobian(self.__fitpars.warp_pars_from_fit_pars(pars)))
        return self.__fitpars.fit_pars_from_warp_pars(wr)

    def _maxTangentialDistortAmountForConstraint(self,pars) :
        """
        call the warp's max tangential distort amount function with the corrected parameters
        """
        return self.__warp.maxTangentialDistortAmount(self.__fitpars.warp_pars_from_fit_pars(pars))

    def _maxTangentialDistortAmountJacobianForConstraint(self,pars) :
        """
        get the correctly-formatted Jacobian of the maximum tangential distortion amount constraint 
        """
        wr = np.array(self.__warp.maxTangentialDistortAmountJacobian(self.__fitpars.warp_pars_from_fit_pars(pars)))
        return self.__fitpars.fit_pars_from_warp_pars(wr)

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __run_differential_evolution(self,max_iters) :
        """
        Actually run differential evolution to determine the best warp and return the result

        max_iters = the maximum number of iterations to allow
        """
        #set up the parameter bounds and constraints and get the initial population
        parameter_bounds = self.__fitpars.get_fit_parameter_bounds()
        constraints = self.__get_constraints()
        initial_population = self.__fitpars.get_initial_population()
        #run the minimization
        self.logger.info('Starting minimization....')
        result=scipy.optimize.differential_evolution(
            func=self._minimization_function,
            bounds=parameter_bounds,
            strategy='best1bin',
            maxiter=int(max_iters/len(initial_population))+1,
            tol=self.DE_TOLERANCE,
            mutation=self.DE_MUTATION,
            recombination=self.DE_RECOMBINATION,
            polish=False, 
            init=initial_population,
            constraints=constraints
            )
        msg = f'Minimization completed {"successfully" if result.success else "UNSUCCESSFULLY"} in '
        msg+= f'{result.nfev} evaluations.'
        self.logger.info(msg)
        return result

    def __get_fit_cost(self) :
        """
        Return the fit cost using the currently-registered WarpingSample overlap images
        """
        weighted_sum_mse = 0.
        sum_weights = 0.
        for io,overlap in enumerate(self.__overlaps,start=1) :
            result = self.__warpsample.align_overlap(overlap)
            if result is not None and result.exit == 0: 
                w=overlap.overlap_npix
                weighted_sum_mse+=w*result.mse[2]
                sum_weights+=w
            else :
                if result is None:
                    reason = "is None"
                else:
                    reason = f"has exit status {result.exit}"
                self.logger.debug(f'Overlap number {overlap.n} alignment result {reason}: returning 1e10!!')
                return 1e10
        return weighted_sum_mse/sum_weights

    def __get_warped_images_by_rect_i(self) :
        """
        Return a dictionary of warped image arrays keyed by rectangle index
        """
        to_return = {}
        for ri,image in self.__unwarped_images_by_rect_i.items() :
            warped_image = self.__warp.getWarpedLayer(image)
            to_return[ri] = warped_image
        return to_return

    def __get_constraints(self) :
        """
        Return the constraint objects for differential evolution
        """
        constraints = []; names_to_print = []
        #if the radial warp is being fit, and the max_radial_warp is defined, add the max_radial_warp constraint
        if self.__fitpars.radial_warp_floating and self.__fitpars.max_rad_warp>0 :
            constraints.append(scipy.optimize.NonlinearConstraint(
                self._maxRadialDistortAmountForConstraint,
                -1.*self.__fitpars.max_rad_warp,
                self.__fitpars.max_rad_warp,
                jac=self._maxRadialDistortAmountJacobianForConstraint,
                hess=scipy.optimize.BFGS()
                )
            )
            names_to_print.append(f'max radial warp={self.__fitpars.max_rad_warp} pixels')
        #if the tangential warping is being fit, and the max_tangential_warp is defined, add the max_tangential_warp constraint
        if self.__fitpars.tangential_warp_floating and self.__fitpars.max_tan_warp>0 :
            constraints.append(scipy.optimize.NonlinearConstraint(
                self._maxTangentialDistortAmountForConstraint,
                0.,
                self.__fitpars.max_tan_warp,
                jac=self._maxTangentialDistortAmountJacobianForConstraint,
                hess=scipy.optimize.BFGS()
                )
            )
            names_to_print.append(f'max tangential warp={self.__fitpars.max_tan_warp} pixels')
        #print the information about the constraints
        if len(constraints)==0 :
            self.logger.info('No constraints will be applied')
            return ()
        else :
            constraintstring = 'Will apply constraints: '
            for ntp in names_to_print :
                constraintstring+=ntp+', '
            self.logger.info(constraintstring[:-2]+'.')
        #return the list of constraints
        if len(constraints)==1 : #if there's only one it doesn't get passed as a list (thanks scipy)
            return constraints[0]
        else :
            return constraints
    