#imports
from .plotting import plot_image_layers
from .latexsummary import MeanImageLatexSummary
from .utilities import FieldLog
from .config import CONST 
from ..image_masking.image_mask import ImageMask
from ..image_masking.utilities import LabelledMaskRegion
from ...shared.logging import dummylogger
from ...utilities.img_file_io import write_image_to_file, smooth_image_worker, smooth_image_with_uncertainty_worker
from ...utilities.tableio import readtable,writetable
from ...utilities.misc import cd
from ...utilities.config import CONST as UNIV_CONST
from multiprocessing.pool import Pool
import numpy as np
import pathlib

class MeanImage :
    """
    Class representing an image that is the mean of a bunch of stacked raw images 
    Includes tools to build meanimages in a variety of ways including masking out empty background and artifacts in images as they're stacked
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,logger=dummylogger) :
        """
        logger = the logging object to use (passed from whatever is using this meanimage)
        """
        self.__logger = logger
        self.__image_stack = None
        self.__image_squared_stack = None
        self.__mask_stack = None
        self.__n_images_read = 0
        self.__n_images_stacked_by_layer = None
        self.__mean_image=None
        self.__std_err_of_mean_image=None

    def stack_images(self,rectangles,med_ets=None,maskingdirpath=None) :
        """
        Loop over a set of given images and add them to the stack
        If masking is being applied, also read each image's masking file before adding it to the stack
        
        rectangles     = the list of rectangles whose images should be added to the stack
        med_ets        = the list of median exposure times by layer for the whole sample, used to normalize images before stacking them
                         if this is None (because the images haven't been corrected for exposure time differences) then the individual 
                         image exposure times will be used instead
        maskingdirpath = the path to the directory holding all of the slide's image masking files (if None then masking will be skipped)
        """
        self.__logger.info(f'Stacking {len(rectangles)} images in the bulk of the tissue')
        #initialize the image and mask stacks based on the image dimensions
        img_dims = rectangles[0].imageshapeinoutput
        self.__image_stack = np.zeros(img_dims,dtype=np.float64)
        self.__image_squared_stack = np.zeros(img_dims,dtype=np.float64)
        if maskingdirpath is not None :
            self.__mask_stack = np.zeros(img_dims,dtype=np.uint64)
        self.__n_images_stacked_by_layer = np.zeros((img_dims[-1]),dtype=np.uint64)
        #expand the median exposure times to apply them in multiple layers at once
        if med_ets is not None :
            med_ets = med_ets[np.newaxis,np.newaxis,:]
        #if the images aren't going to be masked, this whole step is pretty trivial. Otherwise it's a bit more complex
        field_logs_to_return = None
        if maskingdirpath is None :
            self.__logger.info(f'Images will NOT be masked before stacking')
            field_logs_to_return = self.__stack_images_no_masking(rectangles,med_ets)
        else :
            self.__logger.info(f'Images WILL be masked before stacking')
            field_logs_to_return = self.__stack_images_with_masking(rectangles,med_ets,maskingdirpath)
        #make the mean image & its standard error
        self.__logger.info(f'Creating the mean image with its standard error....')
        self.__make_mean_image()
        #return the field logs
        return field_logs_to_return

    def write_output(self,slide_id,workingdirpath) :
        """
        Write out a bunch of information for this meanimage including the meanimage/std. err. files, mask stacks if appropriate, etc.
        Plus also a .pdf showing all of the image layers together

        slide_id       = the ID of the slide to add to the filenames
        workingdirpath = path to the directory where the images etc. should be saved
        """
        self.__logger.info(f'Writing out the mean image, std. err., mask stack....')
        #write out the mean image, std. err. image, and mask stack
        with cd(workingdirpath) :
            write_image_to_file(self.__mean_image,f'{slide_id}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            write_image_to_file(self.__image_squared_stack,f'{slide_id}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}')
            write_image_to_file(self.__std_err_of_mean_image,f'{slide_id}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            if self.__mask_stack is not None :
                write_image_to_file(self.__mask_stack,f'{slide_id}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}')
        self.__logger.info(f'Making plots of image layers and collecting them in the summary pdf....')
        #save .pngs of the mean image/mask stack etc. layers
        plotdir_path = workingdirpath / CONST.MEANIMAGE_SUMMARY_PDF_FILENAME.replace('.pdf','_plots')
        plot_image_layers(self.__mean_image,f'{slide_id}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}'.rstrip('.bin'),plotdir_path)
        plot_image_layers(self.__std_err_of_mean_image,f'{slide_id}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}'.rstrip('.bin'),plotdir_path)
        if self.__mask_stack is not None :
            plot_image_layers(self.__mask_stack,f'{slide_id}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}'.rstrip('.bin'),plotdir_path)
        #collect the plots that were just saved in a .pdf file from a LatexSummary
        latex_summary = MeanImageLatexSummary(slide_id,plotdir_path)
        latex_summary.build_tex_file()
        check = latex_summary.compile()
        if check!=0 :
            warnmsg = f'WARNING: failed while compiling meanimage summary LaTeX file into a PDF. '
            warnmsg+= f'tex file will be in {latex_summary.failed_compilation_tex_file_path}'
            self.__logger.warning(warnmsg)


    #################### PRIVATE HELPER FUNCTIONS ####################

    def __stack_images_no_masking(self,rectangles,med_ets) :
        """
        Simply add all the images to the image_stack and image_squared_stack without masking them
        """
        field_logs = []
        for ri,r in enumerate(rectangles) :
            self.__logger.info(f'Adding {r.file.rstrip(UNIV_CONST.IM3_EXT)} to the meanimage stack ({ri+1} of {len(rectangles)})....')
            with r.using_image() as im :
                normalized_image = im / med_ets if med_ets is not None else im / r.allexposuretimes[np.newaxis,np.newaxis,:]
                self.__image_stack+=normalized_image
                self.__image_squared_stack+=np.power(normalized_image,2)
                self.__n_images_read+=1
                self.__n_images_stacked_by_layer+=1
                field_logs.append(FieldLog(None,r.file.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.RAW_EXT),'bulk','stacking',str(list(range(self.__image_stack.shape[-1])))))
        return field_logs

    def __stack_images_with_masking(self,rectangles,med_ets,maskingdirpath) :
        """
        Read all of the image masks and add the masked images and their masks to the stacks 
        """
        #make sure the masking files exist for every image, otherwise throw a warning and remove the associated image from the list of those to stack
        rectangles_to_stack = rectangles
        if (maskingdirpath/CONST.LABELLED_MASK_REGIONS_CSV_FILENAME).is_file() :
            keys_with_full_masks = set([lmr.image_key for lmr in readtable(maskingdirpath/CONST.LABELLED_MASK_REGIONS_CSV_FILENAME,LabelledMaskRegion)])
        else :
            keys_with_full_masks = set()
        for r in rectangles :
            imkey = r.file.rstrip(UNIV_CONST.IM3_EXT)
            if not (maskingdirpath / f'{imkey}_{CONST.TISSUE_MASK_FILE_NAME_STEM}').is_file() :
                warnmsg = f'WARNING: missing a tissue mask file for {imkey} in {maskingdirpath} and so this image will be skipped!'
                self.__logger.warning(warnmsg)
                rectangles_to_stack.remove(r)
            if imkey in keys_with_full_masks :
                if not (maskingdirpath / f'{imkey}_{CONST.BLUR_AND_SATURATION_MASK_FILE_NAME_STEM}').is_file() :
                    warnmsg = f'WARNING: missing a blur/saturation mask file for {imkey} in {maskingdirpath} and so this image will be skipped!'
                    self.__logger.warning(warnmsg)
                    rectangles_to_stack.remove(r)
        #for every image that will be stacked, read its masking file, normalize and mask its image, and add the image/mask to the respective stacks
        field_logs = []
        for ri,r in enumerate(rectangles_to_stack) :
            self.__logger.info(f'Masking and adding {r.file.rstrip(UNIV_CONST.IM3_EXT)} to the meanimage stack ({ri+1} of {len(rectangles_to_stack)})....')
            imkey = r.file.rstrip(UNIV_CONST.IM3_EXT)
            with r.using_image() as im :
                normalized_im = im / med_ets if med_ets is not None else masked_im / r.allexposuretimes[np.newaxis,np.newaxis,:]
                if imkey in keys_with_full_masks :
                    mask = ImageMask.onehot_mask_from_full_mask_file(maskingdirpath / f'{imkey}_{CONST.BLUR_AND_SATURATION_MASK_FILE_NAME_STEM}',self.__image_stack.shape)
                    layers_to_add = np.where(np.sum(mask,axis=(0,1))/(mask.shape[0]*mask.shape[1])>=CONST.MIN_PIXEL_FRAC,1,0).astype(np.uint64)
                else :
                    mask = (ImageMask.unpack_tissue_mask(maskingdirpath / f'{imkey}_{CONST.TISSUE_MASK_FILE_NAME_STEM}',self.__image_stack.shape[:-1]))[:,:,np.newaxis]
                    layers_to_add = np.ones(im.shape[-1],dtype=np.uint64) if np.sum(mask[:,:,0])/(im.shape[0]*im.shape[1])>=CONST.MIN_PIXEL_FRAC else np.zeros(im.shape[-1],dtype=np.uint64)
                normalized_masked_im = normalized_im*mask*layers_to_add[np.newaxis,np.newaxis,:]
                self.__image_stack+=normalized_masked_im
                self.__image_squared_stack+=np.power(normalized_masked_im,2)
                self.__mask_stack+=mask*layers_to_add[np.newaxis,np.newaxis,:]
                self.__n_images_read+=1
                self.__n_images_stacked_by_layer+=layers_to_add
                stacked_in_layers = [i+1 for i in range(layers_to_add.shape[0]) if layers_to_add[i]==1]
                field_logs.append(FieldLog(None,r.file.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.RAW_EXT),'bulk','stacking',str(stacked_in_layers)))
        return field_logs

    def __make_mean_image(self) :
        """
        Create the mean image with its standard error from the image and mask stacks
        """
        #make sure there actually was at least some number of images used
        if self.__n_images_read<1 or np.max(self.__n_images_stacked_by_layer)<1 :
            if self.__n_images_read<1 :
                msg = f'WARNING: {self.__n_images_read} images were read in and so the mean image will be zero everywhere!'
            else :
                msg = 'WARNING: There are no layers with images stacked in them and so the mean image will be zero everywhere!'    
            self.__logger.warningglobal(msg)
            self.__mean_image = self.__image_stack
            self.__std_err_of_mean_image = self.__image_squared_stack
            return
        #warn if any image layers are missing a substantial number of images in the stack
        for li,nlis in enumerate(self.__n_images_stacked_by_layer,start=1) :
            if nlis<1 :
                msg = f'WARNING: {nlis} images were stacked in layer {li}, so this layer of the meanimage will be zeroes!'
                self.__logger.warningglobal(msg)
        #if the images haven't been masked then this is trivial
        if self.__mask_stack is None :
            self.__mean_image = self.__image_stack/self.__n_images_read
            self.__std_err_of_mean_image = np.sqrt(np.abs(self.__image_squared_stack/self.__n_images_read-(np.power(self.__mean_image,2)))/self.__n_images_read)
            return
        #otherwise though we have to be a bit careful and take the mean value pixel-wise, 
        #being careful to fix any pixels that never got added to so there's no division by zero
        zero_fixed_mask_stack = np.copy(self.__mask_stack)
        zero_fixed_mask_stack[zero_fixed_mask_stack==0] = np.min(zero_fixed_mask_stack[zero_fixed_mask_stack!=0])
        self.__mean_image = self.__image_stack/zero_fixed_mask_stack
        self.__std_err_of_mean_image = np.sqrt(np.abs(self.__image_squared_stack/zero_fixed_mask_stack-(np.power(self.__mean_image,2)))/zero_fixed_mask_stack)

