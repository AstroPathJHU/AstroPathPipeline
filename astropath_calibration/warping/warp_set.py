#imports
from .warp import CameraWarp
from .utilities import warp_logger, WarpingError, loadRawImageWorker, WarpImage
from .config import CONST
from ..utilities.img_file_io import getRawAsHWL
from ..utilities.misc import cd
import numpy as np, multiprocessing as mp
import cv2, contextlib2

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
    @property
    def images_no_corners(self) :
        return [img for img in self.images if not img.is_corner_only] #list of images without corner-only images

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

    def loadRawImages(self,rawfiles,overlaps,rectangles,root_dir,flatfield_file,med_exp_time,et_correction_offset,n_threads=1,smoothsigma=CONST.SMOOTH_SIGMA) :
        """
        Loads files in rawfiles list into a dictionary indexed by filename and layer number to cut down on I/O for repeatedly warping a set of images
        rawfiles             = list of raw, unwarped image filenames (optional, will use value from init if None)
        overlaps             = list of overlaps for this particular fit (optional, only used to mask out images that appear in corner overlaps exclusively)
        rectangles           = list of rectangles for this particular fit (optional, used to streamline updating an AlignmentSet's rectangle images)
        root_dir             = path to Clinical_Specimen directory
        flatfield_file       = path to flatfield file to apply when reading in raw images
        med_exp_time         = the median exposure time for images in this layer of this slide (None if no correction should be applied)
        et_correction_offset = the offset for the exposure time correction for images in this layer of this slide (None if no correction should be applied)
        n_threads            = number of parallel processes to run for reading raw files
        smoothsigma          = sigma for Gaussian smoothing filter applied to raw images on load (set to None to skip smoothing)
        """
        #first load the flatfield corrections
        warp_logger.info(f'Loading flatfield file {flatfield_file} to correct raw image illumination')
        flatfield_layer = (getRawAsHWL(flatfield_file,self.m,self.n,self.nlayers,np.float64))[:,:,self.layer-1] if flatfield_file is not None else None
        if rawfiles is not None :
            self.raw_filenames=rawfiles
        #load the raw images in parallel
        warp_logger.info("Loading raw images...")
        if n_threads> 1 :
            manager = mp.Manager()
            return_dict = manager.dict()
            procs = []
            for i,rf in enumerate(self.raw_filenames,start=1) :
                warp_logger.info(f"    loading {rf} ({i} of {len(self.raw_filenames)}) ...")
                p = mp.Process(target=loadRawImageWorker, 
                               args=(rf,self.m,self.n,self.nlayers,self.layer,
                                     flatfield_layer,med_exp_time,et_correction_offset,
                                     overlaps,rectangles,root_dir,
                                     smoothsigma,return_dict,i-1))
                procs.append(p)
                p.start()
                if len(procs)>=n_threads :
                    for proc in procs :
                        proc.join()
            for proc in procs:
                proc.join()
            for d in return_dict.values() :
                rfkey = d['rfkey']; image = d['image']; is_corner_only = d['is_corner_only']; list_index = d['list_index']
                self.images.append(WarpImage(rfkey,cv2.UMat(image),cv2.UMat(np.empty_like(image)),is_corner_only,list_index))
        #or serially
        else :
            for i,rf in enumerate(self.raw_filenames,start=1) :
                warp_logger.info(f"    loading {rf} ({i} of {len(self.raw_filenames)}) ...")
                d = loadRawImageWorker(rf,self.m,self.n,self.nlayers,self.layer,
                                       flatfield_layer,med_exp_time,et_correction_offset,
                                       overlaps,rectangles,root_dir,
                                       smoothsigma)
                rfkey = d['rfkey']; image = d['image']; is_corner_only = d['is_corner_only']; list_index = d['list_index']
                self.images.append(WarpImage(rfkey,cv2.UMat(image),cv2.UMat(np.empty_like(image)),is_corner_only,list_index))
        #sort them by rectangle index
        self.images.sort(key=lambda x: x.rectangle_list_index)
        #if the list of rectangles was given, make sure that the images are the same length and in the same order
        if rectangles is not None :
            assert len(rectangles)==len(self.images)
        warp_logger.info("Done.")

    def warpLoadedImages(self,skip_corners=False) :
        """
        Warps all the image layers in the raw_images dictionary with the current warp and stores them in memory
        """
        #Warp all the images and store them in memory
        for warpimg in self.images :
            if skip_corners and warpimg.is_corner_only :
                continue
            self.warp.warpLayerInPlace(warpimg.raw_image_umat,warpimg.warped_image_umat)

    def writeOutWarpedImages(self,path=None) :
        """
        Save the warped images as new files in the directory at "path" (or in the current directory if path=None)
        """
        #change to the directory passed in
        try:
            with cd(path) if path is not None else contextlib2.nullcontext():
                #write out all the image files
                for warpimg in self.images :
                    self.warp.writeImageLayer(warpimg.warped_image,warpimg.rawfile_key,self.layer)
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
