#imports
from .warp import CameraWarp
from .utilities import WarpFitResult

class WarpFit :
    """
    A class for performing a single warp fit in a specific way
    """

    def __init__(self,warpsample,octet) :
        """
        warpsample = the WarpingSample object for the given octet (used to get images)
        octet = the OverlapOctet object specifying the relevant rectangles/overlaps
        """
        pass

    def run(self,iters,fixed,init_pars,init_bounds,max_rad_warp,max_tan_warp) :
        """
        Actually run the fit and return its WarpFitResult object
        
        iters = the maximum number of differential evolution iterations to run
        fixed = a list of parameter names that should be kept fixed during fitting
        init_pars = a dictionary of initial parameter values
        init_bounds = a dictionary of initial parameter bounds
        max_rad_warp = the maximum amount of radial warping in pixels to allow (used for a constraint)
        max_tan_warp = the maximum amount of tangential warping in pixels to allow (used for a constraint)
        """
        pass
    