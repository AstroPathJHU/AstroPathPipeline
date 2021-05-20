#imports
import numpy as np

#class for shared constant variables
class Const :
    #final overall outputs
    @property
    def MASK_STACK_DTYPE_OUT(self) :
        return np.uint64 #datatype for the mask stack output image
    @property
    def FILE_EXT(self) :
        return '.bin' #file extension for the main output files
    @property
    def FLATFIELD_FILE_NAME_STEM(self) :
        return 'flatfield' #what the flatfield file is called
    @property
    def MEAN_IMAGE_FILE_NAME_STEM(self) :
        return 'mean_image' #name of the outputted mean image file
    @property
    def SUM_IMAGES_SQUARED_FILE_NAME_STEM(self) :
        return 'sum_images_squared'
    @property
    def STD_ERR_MEAN_IMAGE_FILE_NAME_STEM(self) :
        return 'std_error_of_mean_image' #name of the outputted standard error of the mean image file
    @property
    def MASK_STACK_FILE_NAME_STEM(self) :
        return 'mask_stack' #name of the outputted mask stack file
    @property
    def SMOOTHED_CORRECTED_MEAN_IMAGE_FILE_NAME_STEM(self) :
        return 'smoothed_corrected_mean_image' #name of the outputted smoothed corrected mean image file
    @property
    def THRESHOLDING_PLOT_DIR_NAME(self) :
        return 'thresholding_info' #name of the directory where the thresholding information will be stored
    @property
    def POSTRUN_PLOT_DIRECTORY_NAME(self) :
        return 'postrun_info' #name of directory to hold postrun plots (pixel intensity, image layers, etc.) and other info
    @property
    def IMAGE_LAYER_PLOT_DIRECTORY_NAME(self) :
        return 'image_layer_pngs' #name of directory to hold image layer plots within postrun plot directory
    @property
    def PIXEL_INTENSITY_PLOT_NAME(self) :
        return 'pixel_intensity_plot.png' #name of the pixel intensity plot
    @property
    def N_IMAGES_STACKED_PER_LAYER_PLOT_NAME(self) :
        return 'n_images_stacked_per_layer.png' #name of the images stacked per layer plot
    @property
    def N_IMAGES_READ_TEXT_FILE_NAME(self) :
        return 'n_images_read.txt' #name of the images stacked per layer text file
    @property
    def AUTOMATIC_MEANIMAGE_DIRNAME(self) :
        return 'meanimage'
    @property
    def AUTOMATIC_FLATW_FILE_MEANIMAGE_DIRNAME(self):
        return 'meanimage_from_fw_files'
    @property
    def BATCH_FF_DIRNAME_STEM(self) :
        return 'flatfield_BatchID'
    @property
    def INTENSITY_FIG_WIDTH(self) :
        return 16.8 #width of the intensity plot figure
    @property
    def ILLUMINATION_VARIATION_PLOT_WIDTH(self) :
        return 9.6 #width of the illumination variation plot

CONST=Const()
