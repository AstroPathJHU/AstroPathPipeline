#imports
from .plotting import plot_image_layers, flatfield_image_pixel_intensity_plot, corrected_mean_image_PI_and_IV_plots
from .latexsummary import MeanImageLatexSummary, FlatfieldLatexSummary, AppliedFlatfieldLatexSummary
from .utilities import FieldLog
from .config import CONST
from ..image_masking.image_mask import ImageMask
from ..image_masking.utilities import LabelledMaskRegion
from ...shared.logging import dummylogger
from ...shared.samplemetadata import MetadataSummary
from ...utilities.img_file_io import get_raw_as_hwl, smooth_image_worker, smooth_image_with_uncertainty_worker, write_image_to_file
from ...utilities.tableio import readtable, writetable
from ...utilities.misc import cd
from ...utilities.config import CONST as UNIV_CONST
import numpy as np

class ImageStack :
    """
    Class to handle stacks of images, possibly with masks
    Can be used for making meanimages and flatfield models in a variety of ways
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,img_dims,logger=dummylogger) :
        self.__logger = logger
        self.__image_stack = np.zeros(img_dims,dtype=np.float64)
        self.__image_squared_stack = np.zeros(img_dims,dtype=np.float64)
        self.__mask_stack = None
        self.__mask_stack_inited = False

    def stack_rectangle_images(self,rectangles,med_ets=None,maskingdirpath=None) :
        """
        Loop over a set of given images and add them to the stack
        If masking is being applied, also read each image's masking file before adding it to the stack
        
        rectangles     = the list of rectangles whose images should be added to the stack
        med_ets        = the list of median exposure times by layer for the whole sample that the rectangles are coming from
                         used to normalize images before stacking them
                         if this is None (because the images haven't been corrected for exposure time differences) then the individual 
                         image exposure times will be used instead
        maskingdirpath = the path to the directory holding all of the slide's image masking files (if None then masking will be skipped)
        """
        self.__logger.info(f'Stacking {len(rectangles)} images')
        #initialize the mask stack if necessary
        img_dims = rectangles[0].imageshapeinoutput
        if not self.__mask_stack_inited :
            if maskingdirpath is not None :
                self.__mask_stack = np.zeros(self.__image_stack.shape,dtype=np.uint64)
            self.__mask_stack_inited = True
        if img_dims!=self.__image_stack.shape :
            errmsg = f'ERROR: called stack_images with rectangles that have dimensions {img_dims} but an image stack with dimensions {self.__image_stack.shape}!'
            raise ValueError(errmsg)
        if (maskingdirpath is None)!=(self.__mask_stack is None) :
            raise ValueError('ERROR: cannot mix masked and unmasked images in an ImageStack!')
        #expand the median exposure times to apply them in multiple layers at once
        if med_ets is not None :
            med_ets = med_ets[np.newaxis,np.newaxis,:]
        #if the images aren't going to be masked, this whole step is pretty trivial. Otherwise it's a bit more complex
        if maskingdirpath is None :
            self.__logger.info('Images will NOT be masked before stacking')
            return self.__stack_images_no_masking(rectangles,med_ets)
        else :
            self.__logger.info('Images WILL be masked before stacking')
            return self.__stack_images_with_masking(rectangles,med_ets,maskingdirpath)

    def add_sample_meanimage_from_files(self,sample) :
        """
        Add the already-created meanimage/mask stack for a single given sample to the model by reading its files
        """
        #if it's the first sample we're adding, initialize all of the flatfield's various images based on the sample's dimensions
        if not self.__mask_stack_inited is None :
            self.__mask_stack = np.zeros(self.__image_stack.shape,dtype=np.uint64)
            self.__mask_stack_inited = True
        #make sure the dimensions match
        if sample.rectangles[0].imageshapeinoutput!=self.__image_stack.shape :
            errmsg = f'ERROR: called add_sample_meanimage_from_files with a sample whose rectangles have dimensions {sample.rectangles[0].imageshapeinoutput}'
            errmsg+= f' but an image stack with dimensions {self.__image_stack.shape}'
            raise ValueError(errmsg)
        #read and add the images from the sample
        thismeanimage = get_raw_as_hwl(sample.meanimage,*(self.__image_stack.shape),dtype=np.float64)
        thisimagesquaredstack = get_raw_as_hwl(sample.sumimagessquared,*(self.__image_stack.shape),dtype=np.float64)
        thismaskstack = get_raw_as_hwl(sample.maskstack,*(self.__image_stack.shape),dtype=np.uint64)
        self.__mask_stack+=thismaskstack
        self.__image_stack+=thismaskstack*thismeanimage
        self.__image_squared_stack+=thisimagesquaredstack

    def get_meanimage_and_stderr(self) :
        """
        Create and return the meanimage and its standard error from the image/mask stacks
        DOES NOT check that a sufficient number of images have been stacked; that must be done in some superclass that calls this method
        """
        #if the images haven't been masked then this is trivial
        if self.__mask_stack is None :
            self.__mean_image = self.__image_stack/self.__n_images_read
            self.__std_err_of_mean_image = np.sqrt(np.abs(self.__image_squared_stack/self.__n_images_read-(np.power(self.__mean_image,2)))/self.__n_images_read)
            return
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

    def __stack_images_no_masking(self,rectangles,med_ets) :
        """
        Simply add all the images to the image_stack and image_squared_stack without masking them
        """
        n_images_read = 0
        n_images_stacked_by_layer = np.zeros((rectangles[0].imageshapeinoutput[-1]),dtype=np.uint64)
        field_logs = []
        for ri,r in enumerate(rectangles) :
            self.__logger.info(f'Adding {r.file.rstrip(UNIV_CONST.IM3_EXT)} to the image stack ({ri+1} of {len(rectangles)})....')
            with r.using_image() as im :
                normalized_image = im / med_ets if med_ets is not None else im / r.allexposuretimes[np.newaxis,np.newaxis,:]
                self.__image_stack+=normalized_image
                self.__image_squared_stack+=np.power(normalized_image,2)
                n_images_read+=1
                n_images_stacked_by_layer+=1
                field_logs.append(FieldLog(None,r.file.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.RAW_EXT),'bulk','stacking',str(list(range(self.__image_stack.shape[-1])))))
        return n_images_read, n_images_stacked_by_layer, field_logs

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
        n_images_read = 0
        n_images_stacked_by_layer = np.zeros((rectangles[0].imageshapeinoutput[-1]),dtype=np.uint64)
        field_logs = []
        for ri,r in enumerate(rectangles_to_stack) :
            self.__logger.info(f'Masking and adding {r.file.rstrip(UNIV_CONST.IM3_EXT)} to the image stack ({ri+1} of {len(rectangles_to_stack)})....')
            imkey = r.file.rstrip(UNIV_CONST.IM3_EXT)
            with r.using_image() as im :
                normalized_im = im / med_ets if med_ets is not None else im / r.allexposuretimes[np.newaxis,np.newaxis,:]
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
                n_images_read+=1
                n_images_stacked_by_layer+=layers_to_add
                stacked_in_layers = [i+1 for i in range(layers_to_add.shape[0]) if layers_to_add[i]==1]
                field_logs.append(FieldLog(None,r.file.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.RAW_EXT),'bulk','stacking',str(stacked_in_layers)))
        return n_images_read, n_images_stacked_by_layer, field_logs

class MeanImage(ImageStack) :
    """
    Class representing an image that is the mean of a bunch of stacked raw images 
    Includes tools to build meanimages in a variety of ways including masking out empty background and artifacts in images as they're stacked
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__n_images_read = 0
        self.__n_images_stacked_by_layer = None
        self.__mean_image=None
        self.__std_err_of_mean_image=None

    def stack_rectangle_images(self,rectangles,*otherstackimagesargs) :
        if len(rectangles)<1 :
            return []
        if self.__n_images_stacked_by_layer is None :
            self.__n_images_stacked_by_layer = np.zeros((rectangles[0].imageshapeinoutput[-1]),dtype=np.uint64)
        new_n_images_read, new_n_images_stacked_by_layer, new_field_logs = super().stack_rectangle_images(rectangles,*otherstackimagesargs)
        self.__n_images_read+=new_n_images_read
        self.__n_images_stacked_by_layer+=new_n_images_stacked_by_layer
        return new_field_logs

    def make_mean_image(self) :
        """
        Create the mean image with its standard error from the image and mask stacks
        """
        self.logger.info('Creating the mean image with its standard error....')
        #make sure there actually was at least some number of images used
        if self.__n_images_read<1 or np.max(self.__n_images_stacked_by_layer)<1 :
            if self.__n_images_read<1 :
                msg = f'WARNING: {self.__n_images_read} images were read in and so the mean image will be zero everywhere!'
            else :
                msg = 'WARNING: There are no layers with images stacked in them and so the mean image will be zero everywhere!'    
            self.logger.warningglobal(msg)
            self.__mean_image = self.image_stack
            self.__std_err_of_mean_image = self.image_squared_stack
            return
        #warn if any image layers are missing a substantial number of images in the stack
        for li,nlis in enumerate(self.__n_images_stacked_by_layer,start=1) :
            if nlis<1 :
                msg = f'WARNING: {nlis} images were stacked in layer {li}, so this layer of the meanimage will be zeroes!'
                self.logger.warningglobal(msg)
        self.__mean_image, self.__std_err_of_mean_image = self.get_meanimage_and_stderr()

    def write_output(self,slide_id,workingdirpath) :
        """
        Write out a bunch of information for this meanimage including the meanimage/std. err. files, mask stacks if appropriate, etc.
        Plus also a .pdf showing all of the image layers together

        slide_id       = the ID of the slide to add to the filenames
        workingdirpath = path to the directory where the images etc. should be saved
        """
        self.logger.info('Writing out the mean image, std. err., mask stack....')
        #write out the mean image, std. err. image, and mask stack
        with cd(workingdirpath) :
            write_image_to_file(self.__mean_image,f'{slide_id}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            write_image_to_file(self.image_squared_stack,f'{slide_id}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}')
            write_image_to_file(self.__std_err_of_mean_image,f'{slide_id}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            if self.mask_stack is not None :
                write_image_to_file(self.mask_stack,f'{slide_id}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}')
        self.logger.info('Making plots of image layers and collecting them in the summary pdf....')
        #save .pngs of the mean image/mask stack etc. layers
        plotdir_path = workingdirpath / CONST.MEANIMAGE_SUMMARY_PDF_FILENAME.replace('.pdf','_plots')
        plot_image_layers(self.__mean_image,f'{slide_id}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}'.rstrip('.bin'),plotdir_path)
        plot_image_layers(self.__std_err_of_mean_image,f'{slide_id}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}'.rstrip('.bin'),plotdir_path)
        if self.mask_stack is not None :
            plot_image_layers(self.mask_stack,f'{slide_id}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}'.rstrip('.bin'),plotdir_path)
        #collect the plots that were just saved in a .pdf file from a LatexSummary
        latex_summary = MeanImageLatexSummary(slide_id,plotdir_path)
        latex_summary.build_tex_file()
        check = latex_summary.compile()
        if check!=0 :
            warnmsg = 'WARNING: failed while compiling meanimage summary LaTeX file into a PDF. '
            warnmsg+= f'tex file will be in {latex_summary.failed_compilation_tex_file_path}'
            self.logger.warning(warnmsg)

    #################### PROPERTIES ####################

    @property
    def mean_image(self) :
        return self.__mean_image
    @property
    def std_err_of_mean_image(self) :
        return self.__std_err_of_mean_image

class Flatfield(ImageStack) :
    """
    Class representing the flatfield model for a group of slides
    The model is created by reading the meanimages and other data for each slide
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,**kwargs) :
        """
        logger = the logging object to use (passed from whatever is using this meanimage)
        """
        super().__init__(*args,**kwargs)
        self.__n_images_read = 0
        self.__n_images_stacked_by_layer = None
        self.__flatfield_image = None
        self.__flatfield_image_err = None
        self.__metadata_summaries = []
        self.__field_logs = []

    def add_batchflatfieldsample(self,sample) :
        """
        Add a given sample's information from the meanimage/mask stack/ etc. files to the model
        """
        #use the files to add to the image stack
        self.add_sample_meanimage_from_files(sample)
        #aggregate some metadata as well
        self.__metadata_summaries+=readtable(sample.metadatasummary,MetadataSummary)
        self.__field_logs+=readtable(sample.fieldsused,FieldLog)

    def stack_rectangle_images(self,rectangles,*otherstackimagesargs) :
        if self.__n_images_stacked_by_layer is None :
            self.__n_images_stacked_by_layer = np.zeros((self.image_stack.shape[-1]),dtype=np.uint64)
        new_n_images_read, new_n_images_stacked_by_layer, new_field_logs = super().stack_rectangle_images(rectangles,*otherstackimagesargs)
        self.__n_images_read+=new_n_images_read
        self.__n_images_stacked_by_layer+=new_n_images_stacked_by_layer
        return new_field_logs

    def create_flatfield_model(self) :
        """
        After all the samples have been added, this method creates the actual flatfield object
        """
        self.__flatfield_image = np.empty_like(self.image_stack)
        self.__flatfield_image_err = np.empty_like(self.image_stack)
        #warn if not enough image were stacked overall
        if (self.mask_stack is not None and np.max(self.mask_stack)<1) or (self.mask_stack is None and (self.__n_images_read<1 or np.max(self.__n_images_stacked_by_layer)<1)) :
            self.logger.warningglobal('WARNING: not enough images were stacked overall, so the flatfield model will be all ones!')
            self.__flatfield_image = np.ones_like(self.image_stack)
            self.__flatfield_image_err = np.ones_like(self.image_stack)
            return
        #create the mean image and its standard error from the stacks
        mean_image, std_err_of_mean_image = self.get_meanimage_and_stderr()
        #smooth the mean image with its uncertainty
        smoothed_mean_image,sm_mean_img_err = smooth_image_with_uncertainty_worker(mean_image,std_err_of_mean_image,CONST.FLATFIELD_WIDE_GAUSSIAN_FILTER_SIGMA,gpu=True)
        #create the flatfield image layer-by-layer
        for li in range(self.image_stack.shape[-1]) :
            #warn if the layer is missing a substantial number of images in the stack
            if (self.mask_stack is not None and np.max(self.mask_stack[:,:,li])<1) or (self.mask_stack is None and self.__n_images_stacked_by_layer[li]<1) :
                self.logger.warningglobal(f'WARNING: not enough images were stacked in layer {li+1}, so this layer of the flatfield model will be all ones!')
                self.__flatfield_image[:,:,li] = 1.0
                self.__flatfield_image_err[:,:,li] = 1.0
            else :
                weights = np.divide(np.ones_like(sm_mean_img_err[:,:,li]),(sm_mean_img_err[:,:,li])**2,
                                    out=np.zeros_like(sm_mean_img_err[:,:,li]),where=sm_mean_img_err[:,:,li]>0.)
                if np.sum(weights)<=0 :
                    self.logger.warningglobal(f'WARNING: sum of weights in layer {li+1} is {np.sum(weights)}, so this layer of the flatfield will be all ones!')
                    self.__flatfield_image[:,:,li] = 1.0
                    self.__flatfield_image_err[:,:,li] = 1.0
                else :
                    layermean = np.average(smoothed_mean_image[:,:,li],weights=weights)
                    self.__flatfield_image[:,:,li]=smoothed_mean_image[:,:,li]/layermean
                    self.__flatfield_image_err[:,:,li]=sm_mean_img_err[:,:,li]/layermean

    def write_output(self,batchID,workingdirpath) :
        """
        Write out the flatfield image and all other output

        batchID = the batchID to use for the model (in filenames, etc.)
        workingdirpath = path to the directory where the output should be saved (the actual flatfield is saved in this directory's parent)
        """
        #save the flatfield image and its uncertainty
        if not workingdirpath.is_dir() :
            workingdirpath.mkdir(parents=True)
        with cd(workingdirpath.parent) :
            write_image_to_file(self.__flatfield_image,f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}.bin')
        with cd(workingdirpath) :
            write_image_to_file(self.__flatfield_image_err,f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}_uncertainty.bin')
        #save the metadata summary and the field log
        if len(self.__metadata_summaries)>0 :
            with cd(workingdirpath) :
                writetable(f'{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}',self.__metadata_summaries)
        if len(self.__field_logs)>0 :
            with cd(workingdirpath) :
                writetable(f'{CONST.FIELDS_USED_CSV_FILENAME}',self.__field_logs)
        #make some plots of the image layers and the pixel intensities
        plotdir_path = workingdirpath / f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}_plots'
        plotdir_path.mkdir(exist_ok=True)
        plot_image_layers(self.__flatfield_image,f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}',plotdir_path)
        plot_image_layers(self.__flatfield_image_err,f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}_uncertainty',plotdir_path)
        plot_image_layers(self.mask_stack,f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}_mask_stack',plotdir_path)
        flatfield_image_pixel_intensity_plot(self.__flatfield_image,batchID,plotdir_path)
        #make the summary PDF
        latex_summary = FlatfieldLatexSummary(self.__flatfield_image,plotdir_path,batchID)
        latex_summary.build_tex_file()
        check = latex_summary.compile()
        if check!=0 :
            warnmsg = 'WARNING: failed while compiling flatfield summary LaTeX file into a PDF. '
            warnmsg+= f'tex file will be in {latex_summary.failed_compilation_tex_file_path}'
            self.logger.warning(warnmsg)

    #################### PROPERTIES ####################

    @property
    def flatfield_image(self) :
        return self.__flatfield_image
    @property
    def flatfield_image_err(self) :
        return self.__flatfield_image_err

class CorrectedMeanImage(MeanImage) :
    """
    Class to work with a mean image that will be corrected with a set of flatfield correction factors
    """

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__flatfield_image = None
        self.__flatfield_image_err = None
        self.__corrected_mean_image = None
        self.__corrected_mean_image_err = None

    def apply_flatfield_corrections(self,flatfield) :
        """
        Correct the meanimage using the given flatfield object
        """
        flatfield_image = flatfield.flatfield_image
        flatfield_err = flatfield.flatfield_image_err
        if self.mean_image is None :
            raise RuntimeError('ERROR: mean image is not yet set but apply_flatfield_corrections was called!')
        elif self.mean_image.shape != flatfield_image.shape :
            errmsg = f'ERROR: shape mismatch in apply_flatfield_corrections, meanimage shape = {self.mean_image.shape} but flatfield shape = {flatfield_image.shape}'
            raise ValueError(errmsg)
        self.__flatfield_image = flatfield_image
        self.__flatfield_image_err = flatfield_err
        self.__corrected_mean_image = self.mean_image/flatfield_image
        self.__corrected_mean_image_err = self.__corrected_mean_image*np.sqrt(np.power(self.__flatfield_image_err/self.__flatfield_image,2)*np.power(self.std_err_of_mean_image/self.mean_image,2))

    def write_output(self,workingdirpath) :
        """
        Write out the relevant information for the corrected mean image to the given working directory
        """
        if not workingdirpath.is_dir() :
            workingdirpath.mkdir(parents=True)
        #save the flatfield image and its uncertainty
        self.logger.info('Saving flatfield image and its uncertainty....')
        with cd(workingdirpath) :
            write_image_to_file(self.__flatfield_image,'flatfield.bin')
            write_image_to_file(self.__flatfield_image_err,'flatfield_uncertainty.bin')
        #write out the mask stack
        if self.mask_stack is not None :
            self.logger.info('Writing out mask stack for mean image....')
            with cd(workingdirpath) :
                write_image_to_file(self.mask_stack,'corrected_mean_image_mask_stack.bin')
        #write out the corrected mean image and its uncertainty
        self.logger.info('Writing out the corrected mean image and its uncertainty')
        with cd(workingdirpath) :
            write_image_to_file(self.__corrected_mean_image,'corrected_mean_image.bin')
            write_image_to_file(self.__corrected_mean_image_err,'corrected_mean_image_uncertainty.bin')
        #make some plots of the image layers and the pixel intensities
        self.logger.info('Writing out image layer plots....')
        plotdir_path = workingdirpath / 'corrected_meanimage_plots'
        plotdir_path.mkdir(exist_ok=True)
        plot_image_layers(self.__flatfield_image,'flatfield',plotdir_path)
        plot_image_layers(self.__flatfield_image_err,'flatfield_uncertainty',plotdir_path)
        if self.mask_stack is not None :
            plot_image_layers(self.mask_stack,'corrected_mean_image_mask_stack',plotdir_path)
        plot_image_layers(self.mean_image,'mean_image',plotdir_path)
        plot_image_layers(self.std_err_of_mean_image,'mean_image_uncertainty',plotdir_path)
        plot_image_layers(self.__corrected_mean_image,'corrected_mean_image',plotdir_path)
        plot_image_layers(self.__corrected_mean_image_err,'corrected_mean_image_uncertainty',plotdir_path)
        self.logger.info('Building smoothed mean images pre/post correction....')
        smoothed_mean_image = smooth_image_worker(self.mean_image,CONST.FLATFIELD_WIDE_GAUSSIAN_FILTER_SIGMA,gpu=True)
        smoothed_corrected_mean_image = smooth_image_worker(self.__corrected_mean_image,CONST.FLATFIELD_WIDE_GAUSSIAN_FILTER_SIGMA,gpu=True)
        self.logger.info('Plotting pixel intensities....')
        flatfield_image_pixel_intensity_plot(self.__flatfield_image,save_dirpath=plotdir_path)
        corrected_mean_image_PI_and_IV_plots(smoothed_mean_image,smoothed_corrected_mean_image,central_region=False,save_dirpath=plotdir_path)
        corrected_mean_image_PI_and_IV_plots(smoothed_mean_image,smoothed_corrected_mean_image,central_region=True,save_dirpath=plotdir_path)
        #make the summary PDF
        self.logger.info('Making the summary pdf....')
        latex_summary = AppliedFlatfieldLatexSummary(self.__flatfield_image,smoothed_mean_image,smoothed_corrected_mean_image,plotdir_path)
        latex_summary.build_tex_file()
        check = latex_summary.compile()
        if check!=0 :
            warnmsg = 'WARNING: failed while compiling flatfield summary LaTeX file into a PDF. '
            warnmsg+= f'tex file will be in {latex_summary.failed_compilation_tex_file_path}'
            self.logger.warning(warnmsg)
