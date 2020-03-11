#imports
from .warp import PolyFieldWarp, CameraWarp, WarpingError
import numpy as np
import dataclasses, os, copy, logging

class WarpSet :
    """
    Main class for handling I/O and repeated warp application
    """
    def __init__(self,n=None,m=None,warp=None,rawfiles=None,nlayers=35,layer=1) :
        """
        warp     = warp object to apply to images (optional, default loads a default CameraWarp)
        rawfiles = list of raw, unwarped image filenames 
        nlayers  = # of layers in raw images (default=35)
        layer    = layer number to extract from raw images and warp (indexed from 1, default=1)
        """
        self.warp=warp
        if self.warp is None :
            if n is not None and m is not None :
                self.warp=CameraWarp(n,m)
            else :
                self.warp = CameraWarp()
        self.raw_filenames = rawfiles
        self.nlayers=nlayers
        self.layer=layer
        self.images = []

    #################### COMMON FUNCTIONS ####################

    def loadRawImageSet(self,rawfiles=None,overlaps=None,rectangles=None) :
        """
        Loads files in rawfiles list into a dictionary indexed by filename and layer number to cut down on I/O for repeatedly warping a set of images
        rawfiles   = list of raw, unwarped image filenames (optional, will use value from init if None)
        overlaps   = list of overlaps for this particular fit (optional, only used to mask out images that appear in corner overlaps exclusively)
        rectangles = list of rectangles for this particular fit (optional, only used to mask out images that appear in corner overlaps exclusively)
        """
        logger = logging.getLogger("warpfitter")
        if rawfiles is not None :
            self.raw_filenames=rawfiles
        logger.info("Loading raw images...")
        for i,rf in enumerate(self.raw_filenames,start=1) :
            logger.info(f"    loading {rf} ({i} of {len(self.raw_filenames)}) ...")
            rawimage = self.warp.getHWLFromRaw(rf,self.nlayers)
            rfkey = (rf.split(os.sep)[-1]).split('.')[0]
            #find out if this image should be masked when skipping the corner overlaps
            is_corner_only=False #default is to consider every image
            if overlaps is not None and rectangles is not None :
                this_rect_number = [r.n for r in rectangles if r.file.split('.')[0]==rfkey]
                assert len(this_rect_number)==1
                this_rect_number=this_rect_number[0]
                this_rect_overlap_tags=[o.tag for o in overlaps if o.p1==this_rect_number or o.p2==this_rect_number]
                is_corner_only=True
                for tag in this_rect_overlap_tags :
                    if tag not in [1,3,7,9] :
                        is_corner_only=False
                        break
            new_raw_img=copy.copy(rawimage[:,:,self.layer])
            self.images.append(WarpImage(rfkey,new_raw_img,np.zeros_like(new_raw_img),is_corner_only))
        logger.info("Done.")

    def warpLoadedImageSet(self,skip_corners=False) :
        """
        Warps all the image layers in the raw_images dictionary with the current warp and stores them in memory
        """
        #If we haven't loaded the raw images yet, do it now with the default arguments : /
        if len(self.images)==0 :
            self.loadRawImageSet()
        #Warp all the images and store them in memory
        for warpimg in self.images :
            if skip_corners and warpimg.is_corner_only :
                continue
            self.warp.warpLayerInPlace(warpimg.raw_image,warpimg.warped_image)

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
        #write out all the image files
        for warpimg in self.images :
            self.warp.writeImageLayer(warpimg.warped_image,warpimg.rawfile_key,self.layer)
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

#helper class to hold a rectangle's rawfile key, raw image, warped image, and tag for whether it's only relevant for overlaps that are corners
@dataclasses.dataclass(eq=False, repr=False)
class WarpImage :
    rawfile_key    : str
    raw_image      : np.ndarray
    warped_image   : np.ndarray
    is_corner_only : bool

