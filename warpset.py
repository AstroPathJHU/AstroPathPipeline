#imports
from .warp import PolyFieldWarp, CameraWarp, WarpingError
import os

class WarpSet :
    """
    Main class for handling I/O and repeated warp application
    """
    def __init__(self,warp=None,rawfiles=None,nlayers=35,layers=[1]) :
        """
        warp     = warp object to apply to images (optional, default loads a default CameraWarp)
        rawfiles = list of raw, unwarped image filenames (optional, can manually load or overwrite later)
        nlayers  = # of layers in raw images (default=35)
        layers   = list of layer numbers to extract from raw images and warp (indexed from 1, default=[1])
        """
        self.warp=warp
        if self.warp is None :
            self.warp = CameraWarp()
        self.raw_filenames = rawfiles
        self.raw_images = {}
        if rawfiles is not None :
            self.loadRawImageSet(rawfiles,nlayers,layers)
        self.warped_images = {}

    def loadRawImageSet(self,rawfiles,nlayers=35,layers=[1]) :
        """
        Loads files in rawfiles list into a dictionary indexed by filename and layer number to cut down on I/O for repeatedly warping a set of images
        """
        print("Loading raw images...")
        for i,rf in enumerate(rawfiles,start=1) :
            print(f"    loading {rf} ({i} of {len(rawfiles)}) ...")
            rawimage = self.warp.getHWLFromRaw(rf,nlayers)
            rfkey = rf.split(os.sep)[-1]
            if rfkey not in self.raw_images.keys() :
                self.raw_images[rfkey]={}
            for l in layers :
                self.raw_images[rfkey][l]=rawimage[:,:,l]
        print("Done.")

    def warpLoadedImageSet(self) :
        """
        Warps all the image layers in the raw_images dictionary with the current warp and stores them in memory
        """
        for fn in self.raw_images.keys() :
            if fn not in self.warped_images.keys() :
                self.warped_images[fn] = {}
            for l in self.raw_images[fn].keys() :
                self.warped_images[fn][l] = self.warp.getWarpedLayer(self.raw_images[fn][l])

    def updateCameraParams(self,pars) :
        """
        Updates the parameters for the camerawarp
        """
        if not isinstance(self.warp,CameraWarp) :
            raise WarpingError("ERROR: only call updateCameraParams if the warpset is using a CameraWarp!")
        self.warp.updateParams(pars)

