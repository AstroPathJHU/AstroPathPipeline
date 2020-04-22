#imports
from .warp import CameraWarp, WarpingError
from ..utilities.misc import cd
import numpy as np
import cv2, skimage.filters, skimage.util
import contextlib, dataclasses, os, copy, logging

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

    def loadRawImageSet(self,rawfiles=None,overlaps=None,rectangles=None,smoothsigma=0.75,smoothtruncate=3.3) :
        """
        Loads files in rawfiles list into a dictionary indexed by filename and layer number to cut down on I/O for repeatedly warping a set of images
        rawfiles               = list of raw, unwarped image filenames (optional, will use value from init if None)
        overlaps               = list of overlaps for this particular fit (optional, only used to mask out images that appear in corner overlaps exclusively)
        rectangles             = list of rectangles for this particular fit (optional, only used to mask out images that appear in corner overlaps exclusively)
        smooth[sigma/truncate] = sigma/n sigma truncation for Gaussian smoothing filter applied to raw images on load (defaults give an ~4.95 pixel kernel)
                                 set either parameter to None to skip smoothing entirely and just load the raw images
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
            if smoothsigma is not None and smoothtruncate is not None :
                new_raw_img=skimage.util.img_as_uint(skimage.filters.gaussian(new_raw_img,sigma=smoothsigma,truncate=smoothtruncate,mode='reflect'))
            self.images.append(WarpImage(rfkey,cv2.UMat(new_raw_img),cv2.UMat(np.zeros_like(new_raw_img)),is_corner_only))
        logger.info("Done.")

    def warpLoadedImageSet(self,skip_corners=False) :
        """
        Warps all the image layers in the raw_images dictionary with the current warp and stores them in memory
        """
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
        try:
            with cd(path) if path is not None else contextlib.nullcontext():
                #write out all the image files
                for warpimg in self.images :
                    self.warp.writeImageLayer((warpimg.warped_image).get(),warpimg.rawfile_key,self.layer)
        except FileNotFoundError :
            raise FileNotFoundError(f'path {path} supplied to writeOutWarpedImageSet is not a valid location')

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
        return [self.warp.cx,self.warp.cy,self.warp.fx,self.warp.fy,self.warp.k1,self.warp.k2,self.warp.p1,self.warp.p2,self.warp.k3]

#helper class to hold a rectangle's rawfile key, raw image, warped image, and tag for whether it's only relevant for overlaps that are corners
@dataclasses.dataclass(eq=False, repr=False)
class WarpImage :
    rawfile_key    : str
    raw_image      : cv2.UMat
    warped_image   : cv2.UMat
    is_corner_only : bool

