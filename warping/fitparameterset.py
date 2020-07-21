#imports
from .warp import CameraWarp
from .fitparameter import FitParameter
from .utilities import WarpingError, warp_logger

#class to handle everything to do with the warp fitting parameters
class FitParameterSet :

    #################### PROPERTIES ####################
    @property
    def best_fit_warp_parameters(self) : #the ordered list of the best fit warping parameters
        if self._best_fit_warp_parameters is None :
            raise WarpingError('ERROR: bestFitWarpParameters called while best fit parameters is None!')
        return self._best_fit_warp_parameters
    @property
    def result_text_file_lines(self) : # lines of text to be written out in the report file
        max_r_x, max_r_y = self._getMaxDistanceCoords()
        lines = {
            'n':str(self.n),
            'm':str(self.m),
            'cx':str(self.cx)+('' if par_mask[0] else ' (fixed)'),
            'cy':str(self.cy)+('' if par_mask[1] else ' (fixed)'),
            'fx':str(self.fx)+('' if par_mask[2] else ' (fixed)'),
            'fy':str(self.fy)+('' if par_mask[3] else ' (fixed)'),
            'k1':str(self.k1)+('' if par_mask[4] else ' (fixed)'),
            'k2':str(self.k2)+('' if par_mask[5] else ' (fixed)'),
            'k3':str(self.k3)+('' if par_mask[8] else ' (fixed)'),
            'p1':str(self.p1)+('' if par_mask[6] else ' (fixed)'),
            'p2':str(self.p2)+('' if par_mask[7] else ' (fixed)'),
            'max_r_x_coord':str(max_r_x),
            'max_r_y_coord':str(max_r_y),
            'max_r':str(math.sqrt((max_r_x)**2+(max_r_y)**2)),
            'max_radial_warp':str(self.maxRadialDistortAmount(pars)),
            'max_tangential_warp':str(self.maxTangentialDistortAmount(pars)),
        }
        return lines

    #################### CLASS CONSTANTS ####################

    MICROSCOPE_OBJECTIVE_FOCAL_LENGTH = 40000.                        #focal length of the microscope objective (20mm) in pixels
    FITPAR_NAME_LIST = ['cx','cy','fx','fy','k1','k2','p1','p2','k3'] #ordered list of fit parameters names

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,fixed,normalize,max_radial_warp,max_tangential_warp,warp) :
        """
        fixed = list of names of parameters that will be constant during fitting
        normalize = list of names of parameters that should be numerically rescaled between their bounds for fitting
        max_*ial_warp = maximum amounts of radial/tangential warping allowed
        warp = the warp object this parameter set will be applied to
        """
        #make sure the parameters are going to be relevant to a CameraWarp object
        if not isinstance(warp,CameraWarp) :
            raise WarpingError("ERROR: can only use a FitParameterSet with a CameraWarp!")
        #set the maximum warp amounts
        self.max_rad_warp = max_radial_warp
        self.max_tan_warp = max_tangential_warp
        #get the dictionary of parameter bounds based on the maximum warp amounts
        bounds_dict = self.__buildDefaultParameterBoundsDict(warp)
        #make an ordered list of the fit parameters
        self.all_ordered_fitpars = [FitParameter(fpn,bounds_dict[fpn],fpn in fixed,fpn in normalize) for fpn in self.FITPAR_NAME_LIST]
        #initialize the best fit warp parameters
        self._best_fit_warp_parameters = None

    def getGlobalSetup(self) :
        """
        Return the lists of parameter bounds and constraints and the initial population for the global fit
        """
        pass

    def getPolishingSetup(self,force_float_p1p2,p1p2_lasso_lambda) :
        """
        Return the lists of parameter bounds, constraints, initial values, relative step sizes, parameter tolerance, and gradient tolerance for the polishing fit
        force_float_p1p2  = True if tangential warping parameters should be allowed to float (if they were fixed in the global minimization)
        p1p2_lasso_lambda = numerical value of the LASSO constraint that will be applied to the tangential warping parameters (if they are to float)
        """
        x_tol = 1e-4
        g_tol = 1e-5
        pass

    def setInitialResults(self,diff_ev_result) :
        """
        Record the best-fit values of the parameters from the global minimization
        diff_ev_par_values = the list of raw numerical parameter values from differential evolution 
        """
        pass

    def setFinalResults(self,polishing_result) :
        """
        Record the best-fit values of the parameters after all minimization is complete
        final_par_values = the list of raw numerical parameter values from the final minimization
        """
        pass

    def warpParsFromFitPars(self,fit_par_values) :
        """
        Return a full list of warping parameters given the current numerical values of the fit parameters
        fit_par_values = list of numerical fit parameter values
        """
        pass

    def fitParsFromWarpPars(self,warp_par_values) :
        """
        Return a list of just the numerical fit parameters given the full set of warping parameters
        warp_par_values = list of numerical warp parameter values 
        """
        pass

    def getLassoCost(self,fit_par_values) :
        """
        Return the cost from the LASSO constraint on the tangential warping parameters
        """
        lasso_cost = 0.
        if lasso_param_indices is not None :
            for pindex in lasso_param_indices :
                lasso_cost += lasso_lambda*abs(pars[pindex])
        return lasso_cost

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to make the default list of parameter constraints
    def __buildDefaultParameterBoundsDict(self,warp) :
        bounds = {}
        # cx/cy bounds are +/- 10% of the center point
        bounds['cx']=(0.8*(warp.n/2.),1.2*(warp.n/2.))
        bounds['cy']=(0.8*(warp.m/2.),1.2*(warp.m/2.))
        # fx/fy bounds are +/- 2% of the nominal values 
        bounds['fx']=(0.98*self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
        bounds['fy']=(0.98*self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
        # k1/k2/k3 and p1/p2 bounds are 1.5x those that would produce the max radial and tangential warp, respectively, with all others zero
        # (except k1 can't be negative)
        testpars=[warp.n/2,warp.m/2,self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,self.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,0.,0.,0.,0.,0.]
        maxk1 = self.__findDefaultParameterLimit(4,0.1,self.max_rad_warp,warp.maxRadialDistortAmount,testpars)
        bounds['k1']=(0.,1.5*maxk1)
        maxk2 = self.__findDefaultParameterLimit(5,100,self.max_rad_warp,warp.maxRadialDistortAmount,testpars)
        bounds['k2']=(-1.5*maxk2,1.5*maxk2)
        maxk3 = self.__findDefaultParameterLimit(8,10000,self.max_rad_warp,warp.maxRadialDistortAmount,testpars)
        bounds['k3']=(-1.5*maxk3,1.5*maxk3)
        maxp1 = self.__findDefaultParameterLimit(6,0.001,self.max_tan_warp,warp.maxTangentialDistortAmount,testpars)
        bounds['p1']=(-1.5*maxp1,1.5*maxp1)
        maxp2 = self.__findDefaultParameterLimit(7,0.001,self.max_tan_warp,warp.maxTangentialDistortAmount,testpars)
        bounds['p2']=(-1.5*maxp2,1.5*maxp2)
        return bounds

    #helper function to find the limit on a parameter that produces the maximum warp
    def __findDefaultParameterLimit(self,parindex,parincrement,warplimit,warpamtfunc,testpars) :
        warpamt=0.; testparval=0.
        while warpamt<warplimit :
            testparval+=parincrement
            testpars[parindex]=testparval
            warpamt=warpamtfunc(testpars)
        return testparval

    #################### NOT YET EDITED ####################

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
            
    #helper function to get the indices of the LASSO-constrained parameters (necessary??)
    def __getLassoParameterIndices(self,float_p1p2,lasso_lambda) :
        #get the indices of the p1 and p2 parameters to lasso if those are floating
        lasso_indices = None
        if float_p1p2 and lasso_lambda!=0. :
            relevant_parnamelist = (np.array(self.FITPAR_NAME_LIST)[self.par_mask]).tolist()
            lasso_indices = (relevant_parnamelist.index('p1'),relevant_parnamelist.index('p2'))

    #helper function to get the initial parameter values and relative step sizes for the polishing minimization
    def __getPolishingInitialParametersAndRelativeSteps(self) :
        #figure out the initial parameters and their step sizes
        init_pars_to_use = ((np.array(init_pars))[self.par_mask]).tolist()
        for i in range(len(init_pars_to_use)) :
            if init_pars_to_use[i]==0. :
                init_pars_to_use[i]=0.00000001 # can't send p1/p2=0. to the Jacobian functions in trust-constr
        relative_steps = np.array([abs(0.05*p) if abs(p)<1. else 0.05 for p in init_pars_to_use])

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