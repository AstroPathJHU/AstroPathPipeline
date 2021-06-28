#imports
from .plotting import plot_image_layers, flatfield_image_pixel_intensity_plot
from .latexsummary import FlatfieldLatexSummary
from .utilities import FieldLog
from .config import CONST
from ...shared.logging import dummylogger
from ...utilities.img_file_io import get_raw_as_hwl, smooth_image_with_uncertainty_worker, write_image_to_file
from ...utilities.tableio import readtable, writetable
from ...utilities.misc import cd, MetadataSummary
import numpy as np

class Flatfield :
    """
    Class representing the flatfield model for a group of slides
    The model is created by reading the meanimages and other data for each slide
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
        self.__flatfield_image = None
        self.__flatfield_image_err = None
        self.__metadata_summaries = []
        self.__field_logs = []

    def add_sample(self,sample) :
        """
        Add a given sample's information to the model
        """
        #if it's the first sample we're adding, initialize all of the flatfield's various images based on the sample's dimensions
        if self.__image_stack is None :
            self.__image_stack = np.zeros(sample.rectangles[0].imageshapeinoutput,dtype=np.float64)
            self.__image_squared_stack = np.zeros(sample.rectangles[0].imageshapeinoutput,dtype=np.float64)
            self.__mask_stack = np.zeros(sample.rectangles[0].imageshapeinoutput,dtype=np.uint64)
        #read and add the images from the sample
        thismeanimage = get_raw_as_hwl(sample.meanimage,*(self.__image_stack.shape),dtype=np.float64)
        thisimagesquaredstack = get_raw_as_hwl(sample.sumimagessquared,*(self.__image_stack.shape),dtype=np.float64)
        thismaskstack = get_raw_as_hwl(sample.maskstack,*(self.__image_stack.shape),dtype=np.uint64)
        self.__mask_stack+=thismaskstack
        self.__image_stack+=thismaskstack*thismeanimage
        self.__image_squared_stack+=thisimagesquaredstack
        #aggregate some metadata as well
        self.__metadata_summaries+=readtable(sample.metadatasummary,MetadataSummary)
        self.__field_logs+=readtable(sample.fieldsused,FieldLog)

    def create_flatfield_model(self) :
        """
        After all the samples have been added, this method creates the actual flatfield object
        """
        self.__flatfield_image = np.empty_like(self.__image_stack)
        self.__flatfield_image_err = np.empty_like(self.__image_stack)
        #warn if not enough image were stacked overall
        if np.max(self.__mask_stack)<1 :
            self.__logger.warningglobal(f'WARNING: not enough images were stacked overall, so the flatfield model will be all ones!')
            self.__flatfield_image = np.ones_like(self.__image_stack)
            self.__flatfield_image_err = np.ones_like(self.__image_stack)
            return
        #create the mean image and its standard error from the stacks
        zero_fixed_mask_stack = np.copy(self.__mask_stack)
        zero_fixed_mask_stack[zero_fixed_mask_stack==0] = np.min(zero_fixed_mask_stack[zero_fixed_mask_stack!=0])
        mean_image = self.__image_stack/zero_fixed_mask_stack
        std_err_of_mean_image = np.sqrt(np.abs(self.__image_squared_stack/zero_fixed_mask_stack-(np.power(mean_image,2)))/zero_fixed_mask_stack)
        #smooth the mean image with its uncertainty
        smoothed_mean_image,sm_mean_img_err = smooth_image_with_uncertainty_worker(mean_image,std_err_of_mean_image,100)
        #create the flatfield image layer-by-layer
        for li in range(self.__mask_stack.shape[-1]) :
            #warn if the layer is missing a substantial number of images in the stack
            if np.max(self.__mask_stack[:,:,li])<1 :
                self.__logger.warningglobal(f'WARNING: not enough images were stacked in layer {li+1}, so this layer of the flatfield model will be all ones!')
                self.__flatfield_image[:,:,li] = 1.0
                self.__flatfield_image_err[:,:,li] = 1.0
            else :
                weights = np.divide(np.ones_like(sm_mean_img_err[:,:,li]),(sm_mean_img_err[:,:,li])**2,
                                    out=np.zeros_like(sm_mean_img_err[:,:,li]),where=sm_mean_img_err[:,:,li]>0.)
                if np.sum(weights)<=0 :
                    self.__logger.warningglobal(f'WARNING: sum of weights in layer {li+1} is {np.sum(weights)}, so this layer of the flatfield will be all ones!')
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
                writetable(f'{CONST.FIELDS_USED_CSV_FILENAME}.csv',
                           self.__field_logs)
        #make some plots of the image layers and the pixel intensities
        plotdir_path = workingdirpath / f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}_plots'
        plotdir_path.mkdir(exist_ok=True)
        plot_image_layers(self.__flatfield_image,f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}',plotdir_path)
        plot_image_layers(self.__flatfield_image_err,f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}_uncertainty',plotdir_path)
        plot_image_layers(self.__mask_stack,f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}_mask_stack',plotdir_path)
        flatfield_image_pixel_intensity_plot(self.__flatfield_image,batchID,plotdir_path)
        #make the summary PDF
        latex_summary = FlatfieldLatexSummary(self.__flatfield_image,plotdir_path,batchID)
        latex_summary.build_tex_file()
        #check = latex_summary.compile()
        #if check!=0 :
        #    warnmsg = f'WARNING: failed while compiling flatfield summary LaTeX file into a PDF. '
        #    warnmsg+= f'tex file will be in {latex_summary.failed_compilation_tex_file_path}'
        #    self.__logger.warning(warnmsg)
