#imports
from .warp import CameraWarp
from .fitparameter import FitParameter
from .utilities import WarpingError, warp_logger, buildDefaultParameterBoundsDict
import copy, methodtools

#class to handle everything to do with the warp fitting parameters
class FitParameterSet :

    #################### PROPERTIES ####################
    
    @property
    @methodtools.lru_cache()
    def fixed_parameters(self) : #subset of fit parameters that are fixed
        return list([p for p in self.fit_parameters if p.fixed])
    @property
    @methodtools.lru_cache()
    def floating_parameters(self) : #subset of fit parameters that are floating
        return list([p for p in self.fit_parameters if not p.fixed])
    @property
    @methodtools.lru_cache()
    def radial_warp_floating(self): # a boolean of whether the radial warping is floating
        floating_par_names = [p.name for p in self.floating_parameters]
        return ('k1' in floating_par_names and 'k2' in floating_par_names and 'k3' in floating_par_names)
    @property
    @methodtools.lru_cache()
    def tangential_warp_floating(self): # a boolean of whether the tangential warping is floating
        floating_par_names = [p.name for p in self.floating_parameters]
        return ('p1' in floating_par_names and 'p2' in floating_par_names)
    @property
    def best_fit_warp_parameters(self) : #the ordered list of the best fit warping parameters
        if self._best_fit_warp_parameters is None :
            raise WarpingError('ERROR: bestFitWarpParameters called while best fit parameters is None!')
        return self._best_fit_warp_parameters
    @property
    def result_text_file_lines(self) : # lines of text to be written out in the report file
        lines = {}
        for p in self._best_fit_warp_parameters :
            lines[p.name] = str(p.value)+(' (fixed)' if p.fixed else '')
        return lines

    #################### CLASS CONSTANTS ####################

    FIT_PAR_NAME_LIST  = ['cx','cy','fx','fy','k1','k2','k3','p1','p2'] #list of fit parameter names

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,fixed,normalize,max_radial_warp,max_tangential_warp,warp) :
        """
        fixed         = list of names of parameters that will be constant during fitting
        normalize     = list of names of parameters that should be numerically rescaled between their bounds for fitting
        max_*ial_warp = maximum amounts of radial/tangential warping allowed
        warp          = the warp object this parameter set will be applied to
        """
        #make sure the parameters are going to be relevant to a CameraWarp object
        if not isinstance(warp,CameraWarp) :
            raise WarpingError("ERROR: can only use a FitParameterSet with a CameraWarp!")
        #set the maximum warp amounts (always in units of pixels)
        self.max_rad_warp = max_radial_warp
        self.max_tan_warp = max_tangential_warp
        #get the dictionary of absolute parameter bounds based on the maximum warp amounts
        bounds_dict = buildDefaultParameterBoundsDict(warp,self.max_rad_warp,self.max_tan_warp)
        #make an ordered list of the fit parameters
        self.fit_parameters = [FitParameter(fpn,fpn in fixed,fpn in normalize,bounds_dict[fpn],warp.parValueFromName(fpn)) for fpn in self.FIT_PAR_NAME_LIST]
        #initialize the best fit warp parameters
        self._best_fit_warp_parameters = None
        #initialize the p1 and p2 lasso constraint lambda value
        self._p1p2_lasso_lambda = None

    def getFitParameterBounds(self) :
        """
        Return a list of the floating parameter bounds in fit units (also print some information)
        """
        fixed_par_string=''
        for fp in self.fixed_parameters :
            fixed_par_string+=fp.name+', '
        if fixed_par_string!='' :
            msg+=f' ({fixed_par_string[:-2]} fixed)' 
        msg+='.'
        warp_logger.info(msg)
        #print info about the floating parameter bounds
        msg = f'Will fit with {len(self.floating_parameters)} parameters. Bounds (warping units) (fitting units) : \n'
        for p in self.floating_parameters :
            msg+=f'{p.name}     {p.warp_bound_string}     {p.fit_bound_string}\n'
        warp_logger.info(msg)
        #return the list of bounds tuples
        return list([p.fit_bounds for p in self.floating_parameters])

    def getInitialPopulation(self) :
        """
        Return a list of the initial population parameter vectors in fit units
        The initial population is a bunch of linear spaces between the bounds for the parameters individually and in groups
        """
        #make a list of each parameter's grid of possible values
        par_variations = []
        nperpar=5
        for p in self.floating_parameters :
            #principal points and focal lengths are evenly spaced between their bounds
            if p.name in ['cx','cy','fx','fy'] :
                par_variations.append(np.linspace(*(p.fit_bounds),nperpar))
            #other parameters are evenly spaced between (1/2) their bounds
            elif p.name in ['k1','k2','k3','p1','p2'] :
                par_variations.append(np.linspace(*(0.5*b for b in p.fit_bounds),nperpar))
        #first member of the population is the nominal initial parameters
        initial_floating_parameter_values = list([p.initial_fit_value for p in self.floating_parameters])
        population_list = []
        population_list.append(copy.deepcopy(initial_floating_parameter_values))
        #next add a set describing independent parameter variations
        for i in range(len(par_variations)) :
            for val in par_variations[i] :
                if val!=initial_floating_parameter_values[i] :
                    toadd = copy.deepcopy(initial_floating_parameter_values)
                    toadd[i]=val
                    population_list.append(toadd)
        #add sets describing corner limits of groups of parameters
        par_mask = [True if p in self.floating_parameters else False for p in self.fit_parameters]
        parameter_group_sets = [np.array([0,0,1,1,2,2,3,3,2])[par_mask], #parameters that have the same effect
                                np.array([0,1,None,None,0,1,None,None,0])[par_mask], #principal points and radial warps 1                                  
                                np.array([0,1,None,None,1,0,None,None,1])[par_mask], #principal points and radial warps 2
                                np.array([0,1,None,None,0,1,None,None,1])[par_mask], #principal points and radial warps 3                                  
                                np.array([0,1,None,None,1,0,None,None,0])[par_mask], #principal points and radial warps 4
                                np.array([None,None,0,1,0,1,None,None,0])[par_mask], #focal lengths and radial warps 1
                                np.array([None,None,0,1,1,0,None,None,1])[par_mask], #focal lengths and radial warps 2
                                np.array([None,None,0,1,0,1,None,None,1])[par_mask], #focal lengths and radial warps 3
                                np.array([None,None,0,1,1,0,None,None,0])[par_mask], #focal lengths and radial warps 4
                                np.array([0,1,None,None,None,None,0,1,None])[par_mask], #principal points and tangential warps 1
                                np.array([0,1,None,None,None,None,1,0,None])[par_mask], #principal points and tangential warps 2
                                np.array([None,None,0,1,None,None,0,1,None])[par_mask], #focal lengths and tangential warps 1
                                np.array([None,None,0,1,None,None,1,0,None])[par_mask], #focal lengths and tangential warps 2
                                ]
        for parameter_group_numbers in parameter_group_sets :
            group_numbers_to_consider = set(parameter_group_numbers)
            if None in group_numbers_to_consider :
                group_numbers_to_consider.remove(None)
            for group_number in group_numbers_to_consider :
                list_indices = [i for i in range(len(self.floating_parameters)) if parameter_group_numbers[i]==group_number]
                combinations = []
                if len(list_indices)==2 :
                    combinations = [(1,1),(-2,-2),(1,-2),(-2,1),(0,0),(-1,-1),(0,-1),(-1,0)]
                elif len(list_indices)==3 :
                    combinations = [(1,1,1),(-2,-2,1),(1,-2,1),(-2,1,1),(1,1,-2),(-2,-2,-2),(1,-2,-2),(-2,1,-2)]
                for c in combinations :
                    toadd = copy.deepcopy(initial_floating_parameter_values)
                    toadd[list_indices[0]]=0.2*par_variations[list_indices[0]][c[0]]
                    toadd[list_indices[1]]=0.2*par_variations[list_indices[1]][c[1]]
                    if len(list_indices)==3 :
                        toadd[list_indices[2]]=0.2*par_variations[list_indices[2]][c[2]]
                    population_list.append(toadd)
        to_return = np.array(population_list)
        warp_logger.info(f'Initial fit parameter population ({len(population_list)} members):\n{to_return}')
        return to_return

    def getPolishingSetup(self,force_float_p1p2,p1p2_lasso_lambda) :
        """
        Return the lists of parameter bounds, initial values, and relative step sizes for the polishing fit
        force_float_p1p2  = True if tangential warping parameters should be allowed to float (if they were fixed in the global minimization)
        p1p2_lasso_lambda = numerical value of the LASSO constraint that will be applied to the tangential warping parameters (if they are to float)
        """
        #if the tangential warping parameters need to be changed to floating that has to happen first
        if not self.tangential_warp_floating and force_float_p1p2 :
            self.fit_parameters[self.FIT_PAR_NAME_LIST.index('p1')].fixed = False
            self.fit_parameters[self.FIT_PAR_NAME_LIST.index('p2')].fixed = False
        #update the lasso constraint lambda value
        self._p1p2_lasso_lambda = p1p2_lasso_lambda
        #now get the parameter bounds
        bounds = self.getFitParameterBounds()
        #figure out the initial parameters and their step sizes
        initial_parameter_values = list([p.first_minimization_fit_value for p in self.floating_parameters 
                                         if (p.first_minimization_fit_value is not None and p.first_minimization_fit_value!=0) 
                                         else 0.00000001]) # can't send parameter=0. to the Jacobian functions in trust-constr
        relative_steps = np.array([abs(0.03*p) if abs(p)<1. else 0.03 for p in initial_parameter_values])
        #return the information
        return bounds, initial_parameter_values, relative_steps

    def setFirstMinimizationResults(self,diff_ev_result) :
        """
        Record the best-fit values of the parameters from the global minimization
        diff_ev_par_values = the list of raw numerical parameter values from differential evolution 
        """
        for p,fitval in zip(self.floating_parameters,diff_ev_result.x) :
            p.first_minimization_fit_value = fitval

    def setFinalResults(self,polishing_result) :
        """
        Record the best-fit values of the parameters after all minimization is complete
        final_par_values = the list of raw numerical parameter values from the final minimization
        """
        for p,fitval in zip(self.floating_parameters,polishing_result.x) :
            p.second_minimization_fit_value = fitval
        self._best_fit_warp_parameters = copy.deepcopy(self.fit_parameters)

    @methodtools.lru_cache()
    def warpParsFromFitPars(self,fit_par_values) :
        """
        Return a full list of warping parameters given the current numerical values of the fit parameters
        fit_par_values = list of numerical fit parameter values
        """
        to_return = []
        fpi=0
        for p in self.fit_parameters :
            if p.fixed :
                to_return.append(p.current_warp_value)
            else :
                to_return.append(p.warpValueFromFitValue(fit_par_values[fpi]))
                fpi+=1
        return to_return

    @methodtools.lru_cache()
    def fitParsFromWarpPars(self,warp_par_values) :
        """
        Return a list of just the numerical fit parameters given the full set of warping parameters
        warp_par_values = list of numerical warp parameter values 
        """
        return [p.fitValueFromWarpValue(wpv) for p, wpv in zip(self.fit_parameters,warp_par_values) if not p.fixed]

    @methodtools.lru_cache()
    def getLassoCost(self,fit_par_values) :
        """
        Return the cost from the LASSO constraint on the tangential warping parameters
        """
        if self._p1p2_lasso_lambda==0 :
            return 0.
        lasso_cost = 0.
        for pi,p in enumerate(self.floating_parameters) :
            if p.name=='p1' or p.name=='p2' : 
                lasso_cost += self._p1p2_lasso_lambda*abs(fit_par_values[pi])
        return lasso_cost
