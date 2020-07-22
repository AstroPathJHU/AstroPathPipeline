#imports
import methodtools

#class for a single warp fit parameter
class FitParameter :

    #################### PROPERTIES ####################

    @property
    @methodtools.lru_cache()
    def fixed(self): # boolean for whether or not the parameter is fixed
        return self._fixed
    @fixed.setter
    @methodtools.lru_cache()
    def fixed(self,fixed) :
        self._fixed = fixed
    @property
    def warp_bound_string(self) : # a string of the bounds in warp units
        return f'({self._min_warp_val:.2f}, {self._max_warp_val:.2f})'
    @property
    def fit_bound_string(self) : # a string of the bounds in fit units
        return f'({self._min_fit_val:.2f}, {self._max_fit_val:.2f})'
    @property
    def fit_bounds(self) : # a tuple of the bounds in fit units
        return tuple(self._min_fit_val,self._max_fit_val)
    @property
    @methodtools.lru_cache()
    def current_fit_value(self) : #the current value in fit units
        return self._current_fit_value
    @property
    @methodtools.lru_cache()
    def current_warp_value(self) : #the current value in warp units
        return self._current_warp_value
    @property
    def initial_fit_value(self) : #the initial value in fit units
        return self._initial_fit_value
    @property
    def first_minimization_fit_value(self) : #the first minimization best fit value in fit units
        return self._first_minimization_fit_value
    @first_minimization_fit_value.setter
    def first_minimization_fit_value(self,v) :
        self._first_minimization_fit_value = v
        self._first_minimization_warp_value = self.warpValueFromFitValue(v)
        self._current_fit_value = self._first_minimization_fit_value
        self._current_warp_value = self._first_minimization_warp_value
    @property
    def second_minimization_fit_value(self) : #the second minimization best fit value in fit units
        return self._second_minimization_fit_value
    @second_minimization_fit_value.setter
    def second_minimization_fit_value(self,v) :
        self._second_minimization_fit_value = v
        self._second_minimization_warp_value = self.warpValueFromFitValue(v)
        self._current_fit_value = self._second_minimization_fit_value
        self._current_warp_value = self._second_minimization_warp_value

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,name,fixed,normalize,bounds,initial_value) :
        """
        name          = parameter name string
        fixed         = True if parameter will not float in fitting
        normalize     = True if parameter should be numerically rescaled between bounds for fitting
        bounds        = tuple of (lower, upper) absolute parameter bounds in warping units
        initial_value = the starting value of the parameter in warping units
        """
        self.name = name
        self._fixed=fixed
        self._normalize=normalize
        #set the offset and rescale for normalization
        self._offset  = 0.0 if self.min_warp_val==0. else 0.5*(bounds[1]-bounds[0])
        self._rescale = 1.0 if self.min_warp_val==0. else 2.0
        #set the bounds in warping units
        self._min_warp_val = bounds[0]
        self._max_warp_val = bounds[1]
        #set the bounds in fit units
        self._min_fit_val = self.fitValueFromWarpValue(self.min_warp_val)
        self._max_fit_val = self.fitValueFromWarpValue(self.max_warp_val)
        #set the initial numerical values of the parameter
        self._initial_warp_value = initial_value
        self._initial_fit_value  = self.fitValueFromWarpValue(initial_value)
        self._current_warp_value = self._initial_warp_value
        self._current_fit_value = self._initial_fit_value
        #initialize the first and second minimization result values
        self._first_minimization_warp_value = None
        self._first_minimization_fit_value  = None
        self._second_minimization_warp_value = None
        self._second_minimization_fit_value  = None

    @methodtools.lru_cache()
    def fitValueFromWarpValue(self,value) :
        """
        convert a given value from warp units to fit units
        """
        if not self._normalize :
            return value
        else :
            return (value-self._offset)/self._rescale

    @methodtools.lru_cache()
    def warpValueFromFitValue(self,value) :
        """
        convert a given value from fit units to warp units
        """
        if not self._normalize :
            return value
        else :
            return (self._rescale*value)+self._offset
