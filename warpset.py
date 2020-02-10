#imports
from .warp import PolyFieldWarp, CameraWarp, WarpingError

class WarpSet :
    """
    Main class for handling I/O and repeated warp application
    """
    def __init__(self,warp=None,rawfiles=None,nlayers=35,layers=[0]) :
        """
        warp     = warp object to apply to images (optional, default loads a default CameraWarp)
        rawfiles = list of raw, unwarped image filenames (optional, can manually load or overwrite later)
        nlayers  = # of layers in raw images (default=35)
        layers   = list of layer numbers to extract from raw images and warp (indexed from 0, default=[0])
        """
        self.warp=warp
        if self.warp is None :
            self.warp = CameraWarp()
        self.raw_images = {}
        if rawfiles is not None :
            self.loadRawImageSet(rawfiles,nlayers,layers)

    def loadRawImageSet(self,rawfiles,nlayers=35,layers=[0]) :
        """
        Loads files in rawfiles list into a dictionary indexed by filename and layer number to cut down on I/O for repeatedly warping a set of images
        """
        for rf in rawfiles :
            rawimage = self.warp.getHWLFromRaw(rf,nlayers)
            if rf not in self.raw_images.keys() :
                self.raw_images[rf]={}
            for l in layers :
                self.raw_images[rf][l]=rawimage[:,:,l]

    def warpLoadedImageSet(self) :
        """
        Warps all the image layers in the raw_images dictionary with the current warp and saves them
        """
        for fn in self.raw_images.keys() :
            for l in self.raw_images[fn].keys() :
                self.warp.warpLayer(self.raw_images[fn][l],l,fn)

    def updateCameraParams(self,pars) :
        """
        Updates the parameters for the camerawarp
        """
        if not isinstance(self.warp,CameraWarp) :
            raise WarpingError("ERROR: only call updateCameraParams if the warpset is using a CameraWarp!")
        self.warp.updateParams(pars)