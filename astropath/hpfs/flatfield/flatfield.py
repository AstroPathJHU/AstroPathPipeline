#imports
import numpy as np
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.miscfileio import cd
from ...utilities.tableio import readtable, writetable
from ...utilities.img_file_io import smooth_image_with_uncertainty_worker, write_image_to_file
from ...shared.samplemetadata import MetadataSummary
from .config import CONST
from .utilities import FieldLog
from .plotting import plot_image_layers, flatfield_image_pixel_intensity_plot
from .latexsummary import FlatfieldLatexSummary
from .imagestack import ImageStackBase, ImageStackComponentTiff, ImageStackIm3

class FlatfieldBase(ImageStackBase) :
    """
    Base class representing a flatfield model for a group of slides with different image formats
    The model is created by reading the meanimages and other data for each slide
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,**kwargs) :
        """
        logger = the logging object to use (passed from whatever is using this meanimage)
        """
        super().__init__(*args,**kwargs)
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

    def stack_rectangle_images(self,samp,rectangles,*otherstackimagesargs) :
        if self.__n_images_stacked_by_layer is None :
            self.__n_images_stacked_by_layer = np.zeros((self.image_stack.shape[-1]),dtype=np.uint64)
        n_images_read, n_images_stacked_by_layer, field_logs = super().stack_rectangle_images(samp,rectangles,
                                                                                              *otherstackimagesargs)
        self.n_images_read+=n_images_read
        self.__n_images_stacked_by_layer+=n_images_stacked_by_layer
        return field_logs

    def create_flatfield_model(self) :
        """
        After all the samples have been added, this method creates the actual flatfield object
        """
        self.__flatfield_image = np.empty_like(self.image_stack)
        self.__flatfield_image_err = np.empty_like(self.image_stack)
        #warn if not enough image were stacked overall
        if ( (self.mask_stack is not None and np.max(self.mask_stack)<1) or 
             (self.mask_stack is None and (self.n_images_read<1 or np.max(self.__n_images_stacked_by_layer)<1)) ) :
            warnmsg = 'WARNING: not enough images were stacked overall, so the flatfield model will be all ones!'
            self.logger.warningglobal(warnmsg)
            self.__flatfield_image = np.ones_like(self.image_stack)
            self.__flatfield_image_err = np.ones_like(self.image_stack)
            return
        #create the mean image and its standard error from the stacks
        mean_image, std_err_of_mean_image = self.get_meanimage_and_stderr()
        #smooth the mean image with its uncertainty
        sm_mean_image,sm_mean_img_err = smooth_image_with_uncertainty_worker(mean_image,std_err_of_mean_image,
                                                                             CONST.FLATFIELD_WIDE_GAUSSIAN_FILTER_SIGMA,
                                                                             gpu=True)
        #create the flatfield image layer-by-layer
        for li in range(self.image_stack.shape[-1]) :
            #warn if the layer is missing a substantial number of images in the stack
            if ( (self.mask_stack is not None and np.max(self.mask_stack[:,:,li])<1) or 
                 (self.mask_stack is None and self.__n_images_stacked_by_layer[li]<1) ) :
                warnmsg = f'WARNING: not enough images were stacked in layer {li+1}, '
                warnmsg+= 'so this layer of the flatfield model will be all ones!'
                self.logger.warningglobal(warnmsg)
                self.__flatfield_image[:,:,li] = 1.0
                self.__flatfield_image_err[:,:,li] = 1.0
            else :
                weights = np.divide(np.ones_like(sm_mean_img_err[:,:,li]),(sm_mean_img_err[:,:,li])**2,
                                    out=np.zeros_like(sm_mean_img_err[:,:,li]),where=sm_mean_img_err[:,:,li]>0.)
                if np.sum(weights)<=0 :
                    warnmsg = f'WARNING: sum of weights in layer {li+1} is {np.sum(weights)}, '
                    warnmsg+= 'so this layer of the flatfield will be all ones!'
                    self.logger.warningglobal(warnmsg)
                    self.__flatfield_image[:,:,li] = 1.0
                    self.__flatfield_image_err[:,:,li] = 1.0
                else :
                    layermean = np.average(sm_mean_image[:,:,li],weights=weights)
                    self.__flatfield_image[:,:,li]=sm_mean_image[:,:,li]/layermean
                    self.__flatfield_image_err[:,:,li]=sm_mean_img_err[:,:,li]/layermean

    def write_output(self,samp,version,workingdirpath) :
        """
        Write out the flatfield image and all other output

        version = the version to use for the model (in filenames, etc.)
        workingdirpath = path to the directory where the output should be saved 
                         (the actual flatfield is saved in this directory's parent)
        """
        #save the flatfield image and its uncertainty
        if not workingdirpath.is_dir() :
            workingdirpath.mkdir(parents=True)
        with cd(workingdirpath.parent) :
            write_image_to_file(self.__flatfield_image,f'{UNIV_CONST.FLATFIELD_DIRNAME}_{version}.bin')
        with cd(workingdirpath) :
            write_image_to_file(self.__flatfield_image_err,
                                f'{UNIV_CONST.FLATFIELD_DIRNAME}_{version}_uncertainty.bin')
        #save the metadata summary and the field log
        if len(self.__metadata_summaries)>0 :
            with cd(workingdirpath) :
                writetable(f'{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}',self.__metadata_summaries)
        if len(self.__field_logs)>0 :
            with cd(workingdirpath) :
                writetable(f'{CONST.FIELDS_USED_CSV_FILENAME}',self.__field_logs)
        try :
            #make some plots of the image layers and the pixel intensities
            plotdir_path = workingdirpath / f'{UNIV_CONST.FLATFIELD_DIRNAME}_{version}_plots'
            plotdir_path.mkdir(exist_ok=True)
            plot_image_layers(self.__flatfield_image,f'{UNIV_CONST.FLATFIELD_DIRNAME}_{version}',plotdir_path)
            plot_image_layers(self.__flatfield_image_err,
                              f'{UNIV_CONST.FLATFIELD_DIRNAME}_{version}_uncertainty',plotdir_path)
            if self.mask_stack is not None :
                plot_image_layers(self.mask_stack,f'{UNIV_CONST.FLATFIELD_DIRNAME}_{version}_mask_stack',plotdir_path)
            flatfield_image_pixel_intensity_plot(samp,self.__flatfield_image,version,plotdir_path)
            #make the summary PDF
            latex_summary = FlatfieldLatexSummary(self.__flatfield_image,plotdir_path,version)
            latex_summary.build_tex_file()
            check = latex_summary.compile()
            if check!=0 :
                warnmsg = 'WARNING: failed while compiling flatfield summary LaTeX file into a PDF. '
                warnmsg+= f'tex file will be in {latex_summary.failed_compilation_tex_file_path}'
                self.logger.warning(warnmsg)
        except Exception as e :
            warnmsg = 'WARNING: failed to write out some optional plots (scripts will need to be run separately). '
            warnmsg+= f'Exception: {e}'
            self.logger.warning(warnmsg)

    #################### PROPERTIES ####################

    @property
    def flatfield_image(self) :
        return self.__flatfield_image
    @property
    def flatfield_image_err(self) :
        return self.__flatfield_image_err

class FlatfieldComponentTiff(FlatfieldBase,ImageStackComponentTiff) :
    """
    Class representing a flatfield model built from component tiff mean images
    """
    pass

class FlatfieldIm3(FlatfieldBase,ImageStackIm3) :
    """
    Class representing the flatfield model for a group of slides
    The model is created by reading the meanimages and other data for each slide
    """
    pass