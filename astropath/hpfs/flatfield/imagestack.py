#imports
import copy
import numpy as np
from abc import abstractmethod
from contextlib import contextmanager
from threading import Thread
from queue import Queue
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.tableio import readtable
from ...utilities.img_file_io import get_image_hwl_from_xml_file,get_raw_as_hwl
from ...shared.logging import dummylogger, ThingWithLogger
from ...shared.image_masking.utilities import LabelledMaskRegion
from ...shared.image_masking.image_mask import ImageMask
from .config import CONST
from .utilities import FieldLog

class ImageStackBase(ThingWithLogger) :
    """
    Base class for stacks of different types of images, possibly with masks
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,img_dims,logger=dummylogger) :
        self.__logger = logger
        self.__image_stack = np.zeros(img_dims,dtype=np.float64)
        self.__image_squared_stack = np.zeros(img_dims,dtype=np.float64)
        self.__mask_stack = None
        self.n_images_read = 0

    def stack_rectangle_images(self,samp,rectangles,med_ets=None,maskingdirpath=None,nthreads=4) :
        """
        Loop over a set of given images and add them to the stack
        If masking is being applied, also read each image's masking file before adding it to the stack
        
        samp           = the Sample object from which the rectangles originally came
        rectangles     = the list of rectangles whose images should be added to the stack
        med_ets        = the list of median exposure times by layer for the whole sample that the rectangles are 
                         coming from (used to normalize images before stacking them)
                         if this is None (because the images haven't been corrected for exposure time differences) 
                         then the individual image exposure times will be used instead
        maskingdirpath = the path to the directory holding all of the slide's image masking files 
                         (if None then masking will be skipped)
        nthreads       = the number of threads to use in reading image files and their masks to stack them
                         (minimum 4 will be used)
        """
        self.__logger.info(f'Stacking {len(rectangles)} images')
        img_dims = self._get_rectangle_img_shape(rectangles[0])
        if img_dims!=self.__image_stack.shape :
            errmsg = f'ERROR: called stack_images with rectangles that have dimensions {img_dims} '
            errmsg+= f'but an image stack with dimensions {self.__image_stack.shape}!'
            raise ValueError(errmsg)
        #expand the median exposure times to apply them in multiple layers at once
        if med_ets is not None :
            med_ets = med_ets[np.newaxis,np.newaxis,:]
        #if the images aren't going to be masked, this whole step is pretty trivial. Otherwise it's a bit more complex
        if maskingdirpath is None :
            self.__logger.info('Images will NOT be masked before stacking')
            return self.__stack_images_no_masking(rectangles,med_ets,nthreads)
        else :
            self.__logger.info('Images WILL be masked before stacking')
            return self.__stack_images_with_masking(samp,rectangles,med_ets,maskingdirpath,max(nthreads,4))

    def add_sample_meanimage_from_files(self,sample) :
        """
        Add the already-created meanimage/mask stack for a single given sample to the model by reading its files
        """
        #make sure the dimensions match
        if self._get_rectangle_img_shape(sample.rectangles[0])!=self.__image_stack.shape :
            errmsg = 'ERROR: called add_sample_meanimage_from_files with a sample whose rectangles have '
            errmsg+= f'dimensions {self._get_rectangle_img_shape(sample.rectangles[0])} but an image stack with '
            errmsg+= f'dimensions {self.__image_stack.shape}'
            raise ValueError(errmsg)
        #read and add the images from the sample
        thismeanimage = get_raw_as_hwl(sample.meanimage,*(self.__image_stack.shape),dtype=np.float64)
        thisimagesquaredstack = get_raw_as_hwl(sample.sumimagessquared,*(self.__image_stack.shape),dtype=np.float64)
        thismaskstack = get_raw_as_hwl(sample.maskstack,*(self.__image_stack.shape),dtype=np.uint64)
        if self.__mask_stack is None :
            self.__mask_stack = np.zeros(self.__image_stack.shape,dtype=np.uint64)
        self.__mask_stack+=thismaskstack
        self.__image_stack+=thismaskstack*thismeanimage
        self.__image_squared_stack+=thisimagesquaredstack

    def get_meanimage_and_stderr(self) :
        """
        Create and return the meanimage and its standard error from the image/mask stacks
        DOES NOT check that a sufficient number of images have been stacked; 
        that must be done in some superclass that calls this method
        """
        #if the images haven't been masked then this is trivial
        if self.__mask_stack is None :
            self.__mean_image = self.__image_stack/self.n_images_read
            self.__std_err_of_mean_image = np.sqrt(np.abs(self.__image_squared_stack/self.n_images_read-(np.power(self.__mean_image,2)))/self.n_images_read)
            return self.__mean_image, self.__std_err_of_mean_image
        #otherwise though we have to be a bit careful and take the mean value pixel-wise, 
        #being careful to fix any pixels that never got added to so there's no division by zero
        zero_fixed_mask_stack = np.copy(self.__mask_stack)
        zero_fixed_mask_stack[zero_fixed_mask_stack==0] = np.min(zero_fixed_mask_stack[zero_fixed_mask_stack!=0])
        mean_image = self.__image_stack/zero_fixed_mask_stack
        std_err_of_mean_image = np.sqrt(np.abs(self.__image_squared_stack/zero_fixed_mask_stack-(np.power(mean_image,2)))/zero_fixed_mask_stack)
        return mean_image, std_err_of_mean_image

    #################### PROPERTIES ####################

    @property
    def logger(self) :
        return self.__logger
    @logger.setter
    def logger(self,new_logger) :
        self.__logger = new_logger
    @property
    def image_stack(self) :
        return self.__image_stack
    @property
    def image_squared_stack(self) :
        return self.__image_squared_stack
    @property
    def mask_stack(self) :
        return self.__mask_stack

    #################### PRIVATE HELPER FUNCTIONS ####################

    @abstractmethod
    def _get_rectangle_img_shape(self,rect) :
        """
        return the shape of a given rectangle's image. Not implemented in the base class.
        """
        pass

    @staticmethod
    @abstractmethod
    def _add_rect_to_stacking_queue_no_masking(rect,norm_ets,queue) :
        pass

    @staticmethod
    @abstractmethod
    def _add_rect_to_stacking_queue_with_masking(rect,samp,masking_dir_path,keys_with_full_masks,norm_ets,queue) :
        pass

    def __accumulate_from_stacking_queue(self,image_queue,return_queue) :
        n_images_read = 0
        n_images_stacked_by_layer = None
        field_logs = []
        image_queue_item = image_queue.get()
        while image_queue_item is not None :
            if n_images_stacked_by_layer is None :
                n_images_stacked_by_layer = np.zeros((image_queue_item[0].shape[-1]),dtype=np.uint64)
            self.__image_stack+=image_queue_item[0]
            self.__image_squared_stack+=image_queue_item[1]
            n_images_read+=1
            if len(image_queue_item)==3 :
                field_logs.append(copy.copy(image_queue_item[2]))
                n_images_stacked_by_layer+=1
            elif len(image_queue_item)==5 :
                self.__mask_stack+=image_queue_item[2]
                n_images_stacked_by_layer+=image_queue_item[3]
                field_logs.append(copy.copy(image_queue_item[4]))
            image_queue_item = image_queue.get()
        return_queue.put((n_images_read,n_images_stacked_by_layer,field_logs))

    def __stack_images_no_masking(self,rectangles,med_ets,n_threads) :
        """
        Simply add all the images to the image_stack and image_squared_stack without masking them
        """
        image_queue = Queue(n_threads)
        return_queue = Queue()
        nq_threads = []
        acc_thread = Thread(target=self.__accumulate_from_stacking_queue,args=(image_queue,return_queue))
        acc_thread.start()
        for ri,r in enumerate(rectangles) :
            while len(nq_threads)>=(n_threads-1) :
                thread = nq_threads.pop(0)
                thread.join()
            msg = f'Adding {r.file.stem} to the image stack ({ri+1} of {len(rectangles)})....'
            self.__logger.debug(msg)
            new_thread = Thread(target=self._add_rect_to_stacking_queue_no_masking,
                                args=[r,
                                      med_ets if med_ets is not None else r.allexposuretimes[np.newaxis,np.newaxis,:],
                                      image_queue])
            new_thread.start()
            nq_threads.append(new_thread)
        for thread in nq_threads :
            thread.join()
        image_queue.put(None)
        acc_thread.join()
        (n_images_read, n_images_stacked_by_layer, field_logs) = return_queue.get()
        return n_images_read, n_images_stacked_by_layer, field_logs

    def __stack_images_with_masking(self,samp,rectangles,med_ets,maskingdirpath,n_threads) :
        """
        Read all of the image masks and add the masked images and their masks to the stacks 
        """
        #start up the mask stack
        if self.__mask_stack is None :
            self.__mask_stack = np.zeros(self.__image_stack.shape,dtype=np.uint64)
        #make sure the masking files exist for every image, 
        #otherwise throw a warning and remove the associated image from the list of those to stack
        rectangles_to_stack = rectangles
        if (maskingdirpath/CONST.LABELLED_MASK_REGIONS_CSV_FILENAME).is_file() :
            lmrs_as_read = readtable(maskingdirpath/CONST.LABELLED_MASK_REGIONS_CSV_FILENAME,LabelledMaskRegion)
            keys_with_full_masks = set([lmr.image_key for lmr in lmrs_as_read])
        else :
            keys_with_full_masks = set()
        for r in rectangles :
            imkey = r.file.stem
            if not (maskingdirpath / f'{imkey}_{CONST.TISSUE_MASK_FILE_NAME_STEM}').is_file() :
                warnmsg = f'WARNING: missing a tissue mask file for {imkey} in {maskingdirpath} '
                warnmsg+= 'and so this image will be skipped!'
                self.__logger.warning(warnmsg)
                rectangles_to_stack.remove(r)
            if imkey in keys_with_full_masks :
                if not (maskingdirpath / f'{imkey}_{CONST.BLUR_AND_SATURATION_MASK_FILE_NAME_STEM}').is_file() :
                    warnmsg = f'WARNING: missing a blur/saturation mask file for {imkey} in {maskingdirpath} '
                    warnmsg+= 'and so this image will be skipped!'
                    self.__logger.warning(warnmsg)
                    rectangles_to_stack.remove(r)
        #for every image that will be stacked, read its masking file, normalize and mask its image, 
        #and add the masked image/mask to the respective stacks
        image_queue = Queue(n_threads)
        return_queue = Queue()
        nq_threads = []
        acc_thread = Thread(target=self.__accumulate_from_stacking_queue,args=(image_queue,return_queue))
        acc_thread.start()
        for ri,r in enumerate(rectangles_to_stack) :
            while len(nq_threads)>=(n_threads-1) :
                thread = nq_threads.pop(0)
                thread.join()
            msg = f'Masking and adding {r.file.stem} to the image stack '
            msg+= f'({ri+1} of {len(rectangles_to_stack)})....'
            self.__logger.debug(msg)
            new_thread = Thread(target=self._add_rect_to_stacking_queue_with_masking,
                                args=[r,
                                      samp,
                                      maskingdirpath,
                                      keys_with_full_masks,
                                      med_ets if med_ets is not None else r.allexposuretimes[np.newaxis,np.newaxis,:],
                                      image_queue])
            new_thread.start()
            nq_threads.append(new_thread)
        for thread in nq_threads :
            thread.join()
        image_queue.put(None)
        acc_thread.join()
        (n_images_read, n_images_stacked_by_layer, field_logs) = return_queue.get()
        return n_images_read, n_images_stacked_by_layer, field_logs

class ImageStackComponentTiff(ImageStackBase) :
    """
    Class to handle stacks of component tiff images
    """

    def _get_rectangle_img_shape(self,rect) :
        return rect.componenttiffshape

    @staticmethod
    def _add_rect_to_stacking_queue_no_masking(rect,norm_ets,queue) :
        with rect.using_component_tiff() as im :
            normalized_image = im/norm_ets
            new_field_log = FieldLog(None,rect.componenttifffile,
                                     'bulk','stacking',str(list(range(1,normalized_image.shape[-1]+1))))
            queue.put((normalized_image,np.power(normalized_image,2),new_field_log))

    @staticmethod
    def _add_rect_to_stacking_queue_with_masking(rect,samp,masking_dir_path,keys_with_full_masks,norm_ets,queue) :
        imkey = rect.componenttifffile.rstrip(UNIV_CONST.COMPONENT_TIFF_SUFFIX)
        with rect.using_component_tiff() as im :
            normalized_im = im/norm_ets
            #only ever use the tissue masks alone for the component tiffs
            mask_path = masking_dir_path/f'{imkey}_{CONST.TISSUE_MASK_FILE_NAME_STEM}'
            mask = (ImageMask.unpack_tissue_mask(mask_path,im.shape[:-1]))[:,:,np.newaxis]
            layers_to_add = np.ones(im.shape[-1],dtype=np.uint64) if np.sum(mask[:,:,0])/(im.shape[0]*im.shape[1])>=CONST.MIN_PIXEL_FRAC else np.zeros(im.shape[-1],dtype=np.uint64)
            normalized_masked_im = normalized_im*mask*layers_to_add[np.newaxis,np.newaxis,:]
            stacked_in_layers = [i+1 for i in range(layers_to_add.shape[0]) if layers_to_add[i]==1]
            new_field_log = FieldLog(None,rect.componenttifffile,
                                     'bulk','stacking',str(stacked_in_layers))
            queue.put((normalized_masked_im,
                       np.power(normalized_masked_im,2),
                       mask*layers_to_add[np.newaxis,np.newaxis,:],
                       layers_to_add,
                       new_field_log))

class ImageStackIm3(ImageStackBase) :
    """
    Class to handle stacks of raw im3 images
    """

    def _get_rectangle_img_shape(self,rect) :
        return rect.im3shape

    @staticmethod
    def _add_rect_to_stacking_queue_no_masking(rect,norm_ets,queue) :
        with rect.using_corrected_im3() as im :
            normalized_image = im/norm_ets
            new_field_log = FieldLog(None,rect.file.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.RAW_EXT),
                                     'bulk','stacking',str(list(range(1,normalized_image.shape[-1]+1))))
            queue.put((normalized_image,np.power(normalized_image,2),new_field_log))

    @staticmethod
    def _add_rect_to_stacking_queue_with_masking(rect,samp,masking_dir_path,keys_with_full_masks,norm_ets,queue) :
        imkey = rect.file.rstrip(UNIV_CONST.IM3_EXT)
        with rect.using_corrected_im3() as im :
            normalized_im = im/norm_ets
            if imkey in keys_with_full_masks :
                mask_path = masking_dir_path/f'{imkey}_{CONST.BLUR_AND_SATURATION_MASK_FILE_NAME_STEM}'
                mask = ImageMask.onehot_mask_from_full_mask_file_no_blur(samp,mask_path)
                layers_to_add = np.where(np.sum(mask,axis=(0,1))/(mask.shape[0]*mask.shape[1])>=CONST.MIN_PIXEL_FRAC,1,0).astype(np.uint64)
            else :
                mask_path = masking_dir_path/f'{imkey}_{CONST.TISSUE_MASK_FILE_NAME_STEM}'
                mask = (ImageMask.unpack_tissue_mask(mask_path,im.shape[:-1]))[:,:,np.newaxis]
                layers_to_add = np.ones(im.shape[-1],dtype=np.uint64) if np.sum(mask[:,:,0])/(im.shape[0]*im.shape[1])>=CONST.MIN_PIXEL_FRAC else np.zeros(im.shape[-1],dtype=np.uint64)
            normalized_masked_im = normalized_im*mask*layers_to_add[np.newaxis,np.newaxis,:]
            stacked_in_layers = [i+1 for i in range(layers_to_add.shape[0]) if layers_to_add[i]==1]
            new_field_log = FieldLog(None,rect.file.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.RAW_EXT),
                                     'bulk','stacking',str(stacked_in_layers))
            queue.put((normalized_masked_im,
                       np.power(normalized_masked_im,2),
                       mask*layers_to_add[np.newaxis,np.newaxis,:],
                       layers_to_add,
                       new_field_log))
