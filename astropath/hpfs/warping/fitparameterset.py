#imports
import copy
import numpy as np
from .config import CONST
from .utilities import build_default_parameter_bounds_dict
from .warp import CameraWarp
from .fitparameter import FitParameter

#class to handle everything to do with the warp fitting parameters
class FitParameterSet :

    #################### PROPERTIES ####################

    @property
    def fixed_parameters(self) : #subset of fit parameters that are fixed
        return [p for p in self.fit_parameters if p.fixed]
    @property
    def floating_parameters(self) : #subset of fit parameters that are floating
        return [p for p in self.fit_parameters if not p.fixed]
    @property
    def radial_warp_floating(self): # a boolean of whether the radial warping is floating
        floating_par_names = [p.name for p in self.floating_parameters]
        return ('k1' in floating_par_names and 'k2' in floating_par_names and 'k3' in floating_par_names)
    @property
    def tangential_warp_floating(self): # a boolean of whether the tangential warping is floating
        floating_par_names = [p.name for p in self.floating_parameters]
        return ('p1' in floating_par_names and 'p2' in floating_par_names)
    @property
    def best_fit_warp_parameters(self) : #the ordered list of the best fit warping parameters
        if self._best_fit_warp_parameters is None :
            raise RuntimeError('ERROR: bestFitWarpParameters called while best fit parameters is None!')
        return [p.current_warp_value for p in self._best_fit_warp_parameters]

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,fixed,init_pars,init_bounds,max_radial_warp,max_tangential_warp,warp,logger) :
        """
        fixed         = list of names of parameters that will be constant during fitting
        init_pars     = dictionary of initial parameter values; keyed by name
        init_bounds   = dictionary of initial parameter bounds; keyed by name
        max_*ial_warp = maximum amounts of radial/tangential warping allowed
        warp          = the warp object this parameter set will be applied to
        logger        = the logger object to use
        """
        #make sure the parameters are going to be relevant to a CameraWarp object
        if not isinstance(warp,CameraWarp) :
            raise ValueError("ERROR: can only use a FitParameterSet with a CameraWarp!")
        self.logger = logger
        #replace the warp parameters based on the dictionary of initial values
        if init_pars is not None :
            update_pars = [warp.parValueFromName(fpn) for fpn in CONST.ORDERED_FIT_PAR_NAMES]
            for pname,pval in init_pars.items() :
                if pval is not None :
                    if pname not in CONST.ORDERED_FIT_PAR_NAMES :
                        errmsg = f'ERROR: {pname} (initial value={pval}) is not recognized as a fit parameter!'
                        raise ValueError(errmsg)
                    if warp.parValueFromName(pname)!=pval :
                        msg = f'Replacing default {pname} value {warp.parValueFromName(pname)} with {pval}'
                        self.logger.debug(msg)
                        update_pars[CONST.ORDERED_FIT_PAR_NAMES.index(pname)] = pval
            warp.updateParams(update_pars)
        #set the maximum warp amounts (always in units of pixels)
        self.max_rad_warp = max_radial_warp
        self.max_tan_warp = max_tangential_warp
        #get the dictionary of absolute parameter bounds based on the maximum warp amounts and the possible overrides
        bounds_dict = build_default_parameter_bounds_dict(warp,self.max_rad_warp,self.max_tan_warp)
        if init_bounds is not None :
            for pname,pbounds in init_bounds.items() :
                if pbounds is not None :
                    if pname not in CONST.ORDERED_FIT_PAR_NAMES :
                        errmsg = f'ERROR: {pname} (initial bounds={pbounds}) is not recognized as a fit parameter!'
                        raise ValueError(errmsg)
                    if pbounds[0]!=bounds_dict[pname][0] or pbounds[1]!=bounds_dict[pname][1] :
                        self.logger.debug(f'Replacing default {pname} bounds {bounds_dict[pname]} with {pbounds}')
                        if pname in fixed :
                            self.logger.warn(f'WARNING: Replaced bounds for FIXED PARAMETER {pname}!')
                        bounds_dict[pname] = pbounds
        #make an ordered list of the fit parameters
        self.fit_parameters = [FitParameter(fpn,fpn in fixed,bounds_dict[fpn],warp.parValueFromName(fpn)) 
        					   for fpn in CONST.ORDERED_FIT_PAR_NAMES]
        #initialize the best fit warp parameters
        self._best_fit_warp_parameters = None

    def get_fit_parameter_bounds(self) :
        """
        Return a list of the floating parameter bounds in fit units (also print some information)
        """
        msg=f'Will fit with {len(self.floating_parameters)} parameters'
        fixed_par_string=''
        for fp in self.fixed_parameters :
            fixed_par_string+=fp.name+', '
        if fixed_par_string!='' :
            msg+=f' ({fixed_par_string[:-2]} fixed)' 
        msg+='.'
        self.logger.debug(msg)
        #print info about the floating parameter bounds
        lines = []
        lines.append('Bounds (warping_units) (fitting_units):')
        for p in self.floating_parameters :
            lines.append(f'{p.name}     {p.warp_bound_string}     {p.fit_bound_string}')
        spaces = [len('Bounds'),len('(warping_units)'),len('(fitting_units):')]
        for line in lines :
            for fi,field in enumerate(line.split()) :
                if len(field)>spaces[fi] :
                    spaces[fi] = len(field)
        printlns=''
        for line in lines :
            println = ''
            for fi,field in enumerate(line.split()) :
                println+=f'{field:<{spaces[fi]+2}}'
            println+='\n'
            printlns+=println
        for l in printlns.split('\n') :
            self.logger.debug(l)
        #return the list of bounds tuples
        return list([p.fit_bounds for p in self.floating_parameters])

    def get_initial_population(self) :
        """
        Return a list of the initial population parameter vectors in fit units
        The initial population is a bunch of linear spaces between the bounds 
        for the parameters individually and in groups
        """
        #make a list of each parameter's grid of possible values
        par_variations = []
        nperpar=4 if len(self.floating_parameters)<5 else 3
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
        par_group_sets = [np.array([0,0,1,1,2,2,2,3,3])[par_mask], #parameters that have the same effect
                          np.array([0,1,None,None,0,1,0,None,None])[par_mask], #principal points and rad warps 1                                  
                          np.array([0,1,None,None,1,0,1,None,None])[par_mask], #principal points and rad warps 2
                          np.array([0,1,None,None,0,1,1,None,None])[par_mask], #principal points and rad warps 3                                  
                          np.array([0,1,None,None,1,0,0,None,None])[par_mask], #principal points and rad warps 4
                          np.array([None,None,0,1,0,1,0,None,None])[par_mask], #focal lengths and rad warps 1
                          np.array([None,None,0,1,1,0,1,None,None])[par_mask], #focal lengths and rad warps 2
                          np.array([None,None,0,1,0,1,1,None,None])[par_mask], #focal lengths and rad warps 3
                          np.array([None,None,0,1,1,0,0,None,None])[par_mask], #focal lengths and rad warps 4
                          np.array([0,1,None,None,None,None,None,0,1])[par_mask], #principal points and tan warps 1
                          np.array([0,1,None,None,None,None,None,1,0])[par_mask], #principal points and tan warps 2
                          np.array([None,None,0,1,None,None,None,0,1])[par_mask], #focal lengths and tan warps 1
                          np.array([None,None,0,1,None,None,None,1,0])[par_mask], #focal lengths and tan warps 2
                          ]
        for parameter_group_numbers in par_group_sets :
            group_numbers_to_consider = set(parameter_group_numbers)
            if None in group_numbers_to_consider :
                group_numbers_to_consider.remove(None)
            for group_number in group_numbers_to_consider :
                list_indices = [i for i in range(len(self.floating_parameters)) 
                					if parameter_group_numbers[i]==group_number]
                combinations = []
                if len(list_indices)==2 :
                    combinations = [(1,1),(-2,-2),(1,-2),(-2,1),(0,0),(-1,-1),(0,-1),(-1,0)]
                elif len(list_indices)==3 :
                    combinations = [(1,1,1),(-2,-2,1),(1,-2,1),(-2,1,1),(1,1,-2),(-2,-2,-2),(1,-2,-2),(-2,1,-2)]
                for c in combinations :
                    toadd = copy.deepcopy(initial_floating_parameter_values)
                    toadd[list_indices[0]]=par_variations[list_indices[0]][c[0]]
                    toadd[list_indices[1]]=par_variations[list_indices[1]][c[1]]
                    if len(list_indices)==3 :
                        toadd[list_indices[2]]=par_variations[list_indices[2]][c[2]]
                    if toadd not in population_list :
                        population_list.append(toadd)
        to_return = np.array(population_list)
        self.logger.debug(f'Initial fit parameter population has {len(population_list)} members')
        #for pg in population_list :
        #    self.logger.debug(pg)
        return to_return

    def set_final_results(self,diff_ev_result) :
        """
        Record the best-fit values of the parameters after minimization is complete
        diff_ev_result = the differential evolution result object
        """
        for p,fitval in zip(self.floating_parameters,diff_ev_result.x) :
            p.best_fit_value = fitval
        self._best_fit_warp_parameters = copy.deepcopy(self.fit_parameters)

    def warp_pars_from_fit_pars(self,fit_par_values) :
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
                to_return.append(p.warp_value_from_fit_value(fit_par_values[fpi]))
                fpi+=1
        return to_return

    def fit_pars_from_warp_pars(self,warp_par_values) :
        """
        Return a list of just the numerical fit parameters given the full set of warping parameters
        warp_par_values = list of numerical warp parameter values 
        """
        return [p.fit_value_from_warp_value(wpv) for p, wpv in zip(self.fit_parameters,warp_par_values) if not p.fixed]
