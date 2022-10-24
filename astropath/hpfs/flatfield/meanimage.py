#imports
import numpy as np
from ...utilities.miscfileio import cd
from ...utilities.img_file_io import write_image_to_file
from .config import CONST
from .plotting import plot_image_layers
from .latexsummary import MeanImageLatexSummary
from .imagestack import ImageStackBase, ImageStackComponentTiff, ImageStackIm3

class MeanImageBase(ImageStackBase) :
    """
    Base class for any image that is the mean of a bunch of stacked images from some source
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__n_images_stacked_by_layer = None
        self.__mean_image=None
        self.__std_err_of_mean_image=None

    def stack_rectangle_images(self,samp,rectangles,*otherstackimagesargs) :
        if len(rectangles)<1 :
            return []
        if self.__n_images_stacked_by_layer is None :
            self.__n_images_stacked_by_layer = np.zeros((self._get_rectangle_img_shape(rectangles[0])[-1]),dtype=np.uint64)
        n_images_read, n_images_stacked_by_layer, field_logs = super().stack_rectangle_images(samp,rectangles,
                                                                                              *otherstackimagesargs)
        self.n_images_read+=n_images_read
        self.__n_images_stacked_by_layer+=n_images_stacked_by_layer
        return field_logs

    def make_mean_image(self) :
        """
        Create the mean image with its standard error from the image and mask stacks
        """
        self.logger.debug('Creating the mean image with its standard error....')
        #make sure there actually was at least some number of images used
        if self.n_images_read<1 or np.max(self.__n_images_stacked_by_layer)<1 :
            if self.n_images_read<1 :
                msg = f'WARNING: {self.n_images_read} images were read in and so '
                msg+= 'the mean image will be zero everywhere!'
            else :
                msg = 'WARNING: There are no layers with images stacked in them and so '
                msg+= 'the mean image will be zero everywhere!'    
            self.logger.warningglobal(msg)
            self.__mean_image = self.image_stack
            self.__std_err_of_mean_image = self.image_squared_stack
            return
        #warn if any image layers are missing a substantial number of images in the stack
        for li,nlis in enumerate(self.__n_images_stacked_by_layer,start=1) :
            if nlis<1 :
                msg = f'WARNING: {nlis} images were stacked in layer {li}, this layer of the meanimage will be zeroes!'
                self.logger.warningglobal(msg)
        self.__mean_image, self.__std_err_of_mean_image = self.get_meanimage_and_stderr()

    def write_output(self,slide_id,workingdirpath) :
        """
        Write out a bunch of information for this meanimage including the meanimage/std. err. files, 
        mask stacks if appropriate, etc.
        Plus also a .pdf showing all of the image layers together

        slide_id       = the ID of the slide to add to the filenames
        workingdirpath = path to the directory where the images etc. should be saved
        """
        self.logger.debug('Writing out the mean image, std. err., mask stack....')
        #write out the mean image, std. err. image, and mask stack
        with cd(workingdirpath) :
            write_image_to_file(self.__mean_image,f'{slide_id}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            write_image_to_file(self.image_squared_stack,f'{slide_id}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}')
            write_image_to_file(self.__std_err_of_mean_image,
                                f'{slide_id}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            if self.mask_stack is not None :
                write_image_to_file(self.mask_stack,f'{slide_id}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}')
        self.logger.debug('Making plots of image layers and collecting them in the summary pdf....')
        try :
            #save .pngs of the mean image/mask stack etc. layers
            plotdir_path = workingdirpath / CONST.MEANIMAGE_SUMMARY_PDF_FILENAME.replace('.pdf','_plots')
            plot_image_layers(self.__mean_image,
                            f'{slide_id}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}'.rstrip('.bin'),plotdir_path)
            plot_image_layers(self.__std_err_of_mean_image,
                            f'{slide_id}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}'.rstrip('.bin'),plotdir_path)
            if self.mask_stack is not None :
                plot_image_layers(self.mask_stack,
                                f'{slide_id}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}'.rstrip('.bin'),plotdir_path)
            #collect the plots that were just saved in a .pdf file from a LatexSummary
            latex_summary = MeanImageLatexSummary(slide_id,plotdir_path)
            latex_summary.build_tex_file()
            check = latex_summary.compile()
            if check!=0 :
                warnmsg = 'WARNING: failed while compiling meanimage summary LaTeX file into a PDF. '
                warnmsg+= f'tex file will be in {latex_summary.failed_compilation_tex_file_path}'
                self.logger.warning(warnmsg)
        except Exception as e :
            warnmsg = 'WARNING: failed to write out some optional plots (scripts will need to be run separately). '
            warnmsg+= f'Exception: {e}'
            self.logger.warning(warnmsg)

    #################### PROPERTIES ####################

    @property
    def mean_image(self) :
        return self.__mean_image
    @property
    def std_err_of_mean_image(self) :
        return self.__std_err_of_mean_image

class MeanImageComponentTiff(MeanImageBase,ImageStackComponentTiff) :
    """
    Class representing an image that is the mean of a bunch of stacked unmixed component tiff images 
    """
    pass

class MeanImageIm3(MeanImageBase,ImageStackIm3) :
    """
    Class representing an image that is the mean of a bunch of stacked raw images 
    """
    pass
