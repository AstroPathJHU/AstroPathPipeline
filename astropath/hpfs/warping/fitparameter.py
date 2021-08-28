#imports
import methodtools

class FitParameter :
    """
    class for a single warp fit parameter
    """

    #################### PROPERTIES ####################

    @property
    def fixed(self): # boolean for whether or not the parameter is fixed
        return self._fixed
    @fixed.setter
    def fixed(self,fixed) :
        self._fixed = fixed
    @property
    def warp_bound_string(self) : # a string of the bounds in warp units
        return f'({self._min_warp_val:.2f},{self._max_warp_val:.2f})'
    @property
    def fit_bound_string(self) : # a string of the bounds in fit units
        return f'({self._min_fit_val:.2f},{self._max_fit_val:.2f})'
    @property
    def fit_bounds(self) : # a tuple of the bounds in fit units
        return (self._min_fit_val,self._max_fit_val)
    @property
    def current_fit_value(self) : #the current value in fit units
        return self._current_fit_value
    @property
    def current_warp_value(self) : #the current value in warp units
        return self._current_warp_value
    @property
    def initial_fit_value(self) : #the initial value in fit units
        return self._initial_fit_value
    @property
    def best_fit_value(self) : #the best fit value in fit units
        return self._best_fit_value
    @best_fit_value.setter
    def best_fit_value(self,v) :
        self._best_fit_value = v
        self._best_warp_value = self.warp_value_from_fit_value(v)
        self._current_fit_value = self._best_fit_value
        self._current_warp_value = self._best_warp_value

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,name,fixed,bounds,initial_value) :
        """
        name          = parameter name string
        fixed         = True if parameter will not float in fitting
        bounds        = tuple of (lower, upper) absolute parameter bounds in warping units
        initial_value = the starting value of the parameter in warping units
        """
        self.name = name
        self._fixed=fixed
        self._normalize=True
        #set the bounds in warping units
        self._min_warp_val = bounds[0]
        self._max_warp_val = bounds[1]
        #set the offset and rescale for normalization
        self._offset  = 0.0 if self._min_warp_val==0. else 0.5*(bounds[1]+bounds[0])
        self._rescale = (bounds[1]-bounds[0]) if self._min_warp_val==0. else 0.5*(bounds[1]-bounds[0])
        #set the bounds in fit units
        self._min_fit_val = self.fit_value_from_warp_value(self._min_warp_val)
        self._max_fit_val = self.fit_value_from_warp_value(self._max_warp_val)
        #set the initial numerical values of the parameter
        self._initial_warp_value = initial_value
        self._initial_fit_value  = self.fit_value_from_warp_value(initial_value)
        self._current_warp_value = self._initial_warp_value
        self._current_fit_value  = self._initial_fit_value
        #initialize the best fit result values
        self._best_warp_value = None
        self._best_fit_value  = None

    @methodtools.lru_cache()
    def fit_value_from_warp_value(self,value) :
        """
        convert a given value from warp units to fit units
        """
        if not self._normalize :
            return value
        else :
            return (value-self._offset)/self._rescale

    @methodtools.lru_cache()
    def warp_value_from_fit_value(self,value) :
        """
        convert a given value from fit units to warp units
        """
        if not self._normalize :
            return value
        else :
            return (self._rescale*value)+self._offset
