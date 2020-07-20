#imports
from .warp import CameraWarp
from .utilities import warp_logger, WarpingError, loadRawImageWorker, WarpImage
from .config import CONST
from ..utilities.img_file_io import getRawAsHWL
from ..utilities.misc import cd
import numpy as np, multiprocessing as mp
import cv2, contextlib

class WarpSet :
    """
    Main class for handling I/O and repeated warp application
    """

    ####################### PROPERTIES #######################

    @property
    def m(self):
        return self.warp.m #image height
    @property
    def n(self):
        return self.warp.n #image width

    #################### PUBLIC FUNCTIONS ####################

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

    def loadRawImages(self,rawfiles=None,overlaps=None,rectangles=None,flatfield_file_path=None,n_threads=1,smoothsigma=CONST.smoothsigma) :
        """
        Loads files in rawfiles list into a dictionary indexed by filename and layer number to cut down on I/O for repeatedly warping a set of images
        rawfiles            = list of raw, unwarped image filenames (optional, will use value from init if None)
        overlaps            = list of overlaps for this particular fit (optional, only used to mask out images that appear in corner overlaps exclusively)
        rectangles          = list of rectangles for this particular fit (optional, only used to mask out images that appear in corner overlaps exclusively)
        flatfield_file_path = path to flatfield file to apply when reading in raw images
        n_threads           = number of parallel processes to run for reading raw files
        smoothsigma         = sigma for Gaussian smoothing filter applied to raw images on load (set to None to skip smoothing)
        """
        #first load the flatfield corrections
        warp_logger.info(f'Loading flatfield file {flatfield_file_path} to correct raw image illumination')
        if flatfield_file_path is not None :
            flatfield_file_layer = (getRawAsHWL(flatfield_file_path,self.m,self.n,self.nlayers,np.float64))[:,:,self.layer-1] 
        else :
            flatfield_file_layer = np.ones((self.m,self.n),dtype=np.float64)
        if rawfiles is not None :
            self.raw_filenames=rawfiles
        warp_logger.info("Loading raw images...")
        if n_threads> 1 :
            manager = mp.Manager()
            return_dict = manager.dict()
            procs = []
            for i,rf in enumerate(self.raw_filenames,start=1) :
                warp_logger.info(f"    loading {rf} ({i} of {len(self.raw_filenames)}) ...")
                p = mp.Process(target=loadRawImageWorker, 
                               args=(rf,self.m,self.n,self.nlayers,self.layer,flatfield_layer,overlaps,rectangles,smoothsigma,return_dict,i-1))
                procs.append(p)
                p.start()
                if len(procs)>=n_threads :
                    for proc in procs :
                        proc.join()
            for proc in procs:
                proc.join()
            for d in return_dict.values() :
                rfkey = d['rfkey']; image = d['image']; is_corner_only = d['is_corner_only']
                self.images.append(WarpImage(rfkey,cv2.UMat(image),cv2.UMat(np.empty_like(image)),is_corner_only))
        else :
            for i,rf in enumerate(self.raw_filenames,start=1) :
                warp_logger.info(f"    loading {rf} ({i} of {len(self.raw_filenames)}) ...")
                d = loadRawImageWorker(rf,self.m,self.n,self.nlayers,self.layer,flatfield_layer,overlaps,rectangles,smoothsigma)
                rfkey = d['rfkey']; image = d['image']; is_corner_only = d['is_corner_only']
                self.images.append(WarpImage(rfkey,cv2.UMat(image),cv2.UMat(np.empty_like(image)),is_corner_only))
        warp_logger.info("Done.")

    def warpLoadedImages(self,skip_corners=False) :
        """
        Warps all the image layers in the raw_images dictionary with the current warp and stores them in memory
        """
        #Warp all the images and store them in memory
        for warpimg in self.images :
            if skip_corners and warpimg.is_corner_only :
                continue
            self.warp.warpLayerInPlace(warpimg.raw_image,warpimg.warped_image)

    def writeOutWarpedImages(self,path=None) :
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
