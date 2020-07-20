#class for a single warp fit parameter
class FitParameter :

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,name,bounds,fixed,normalize) :
        """
		name      = parameter name string
		bounds    = tuple of (lower, upper) absolute parameter bounds
		fixed     = True if parameter will not float in fitting
		normalize = True if parameter should be numerically rescaled between bounds for fitting
        """
        self.name = name
        self.min = bounds[0]
        self.max = bounds[1]
        self.fixed=fixed
        self.normalize=normalize

    #################### PRIVATE HELPER FUNCTIONS ####################

    