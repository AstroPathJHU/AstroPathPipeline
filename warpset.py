#imports
from .warp import PolyFieldWarp, CameraWarp, WarpingError
import numpy as np
import os, copy, logging

class WarpSet :
    """
    Main class for handling I/O and repeated warp application
    """
    def __init__(self,n=None,m=None,warp=None,rawfiles=None,nlayers=35,layers=[1]) :
        """
        warp     = warp object to apply to images (optional, default loads a default CameraWarp)
        rawfiles = list of raw, unwarped image filenames 
        nlayers  = # of layers in raw images (default=35)
        layers   = list of layer numbers to extract from raw images and warp (indexed from 1, default=[1])
        """
        self.warp=warp
        if self.warp is None :
            if n is not None and m is not None :
                self.warp=CameraWarp(n,m)
            else :
                self.warp = CameraWarp()
        self.raw_filenames = rawfiles
        self.nlayers=nlayers
        self.layers=layers
        self.raw_images = {}
        self.warped_images = {}

    #################### COMMON FUNCTIONS ####################

    def loadRawImageSet(self,rawfiles=None) :
        """
        Loads files in rawfiles list into a dictionary indexed by filename and layer number to cut down on I/O for repeatedly warping a set of images
        rawfiles = list of raw, unwarped image filenames (optional, will use value from init if None)
        """
        logger = logging.getLogger("warpfitter")
        if rawfiles is not None :
            self.raw_filenames=rawfiles
        logger.info("Loading raw images...")
        for i,rf in enumerate(self.raw_filenames,start=1) :
            logger.info(f"    loading {rf} ({i} of {len(self.raw_filenames)}) ...")
            rawimage = self.warp.getHWLFromRaw(rf,self.nlayers)
            rfkey = rf.split(os.sep)[-1]
            if rfkey not in self.raw_images.keys() :
                self.raw_images[rfkey]={}
                self.warped_images[rfkey]={}
            for l in self.layers :
                self.raw_images[rfkey][l]=copy.copy(rawimage[:,:,l])
                self.warped_images[rfkey][l]=np.zeros_like(self.raw_images[rfkey][l])
        logger.info("Done.")

    def warpLoadedImageSet(self) :
        """
        Warps all the image layers in the raw_images dictionary with the current warp and stores them in memory
        """
        #If we haven't loaded the raw images yet, do it now : /
        if self.raw_images=={} :
            self.loadRawImageSet()
        #Warp all the image layers and store them in memory
        for fn in self.raw_images.keys() :
            for l in self.raw_images[fn].keys() :
                self.warp.warpLayerInPlace(self.raw_images[fn][l],self.warped_images[fn][l])

    def writeOutWarpedImageSet(self,path=None) :
        """
        Save the warped images as new files in the directory at "path" (or in the current directory if path=None)
        """
        #change to the directory passed in
        if path is not None :
            init_dir = os.getcwd()
            try :
                os.chdir(path)
            except FileNotFoundError :
                raise FileNotFoundError(f'path {path} supplied to writeOutWarpedImageSet is not a valid location')
        #write out all the image file layers
        for fn in self.raw_images.keys() :
            for l in self.raw_images[fn].keys() :
                self.warp.writeImageLayer(self.warped_images[fn][l],fn,l)
        #change back to the initial directory
        if path is not None :
            os.chdir(init_dir)

    def updateCameraParams(self,pars) :
        """
        Updates the parameters for the camerawarp
        """
        if not isinstance(self.warp,CameraWarp) :
            raise WarpingError("ERROR: only call updateCameraParams if the warpset is using a CameraWarp!")
        self.warp.updateParams(pars)

    def getListOfWarpParameters(self) :
        """
        Returns a list of all the current warp parameters
        """
        if not isinstance(self.warp,CameraWarp) :
            raise WarpingError("ERROR: only call getListOfWarpParameters if the warpset is using a CameraWarp!")
        return [self.warp.cx,self.warp.cy,self.warp.fx,self.warp.fy,self.warp.k1,self.warp.k2,self.warp.p1,self.warp.p2]

