#imports
import numpy as np
import cv2

#class for shared constant variables
class Const :
    #final overall outputs
    @property
    def IMG_DTYPE_OUT(self) :
        return np.float64 #datatype for the main output images
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
    def STD_ERR_MEAN_IMAGE_FILE_NAME_STEM(self) :
        return 'std_error_of_mean_image' #name of the outputted standard error of the mean image file
    @property
    def MASK_STACK_FILE_NAME_STEM(self) :
        return 'mask_stack' #name of the outputted mask stack file
    @property
    def SMOOTHED_CORRECTED_MEAN_IMAGE_FILE_NAME_STEM(self) :
        return 'smoothed_corrected_mean_image' #name of the outputted smoothed corrected mean image file
    @property
    def THRESHOLD_TEXT_FILE_NAME_STEM(self) :
        return 'background_thresholds.txt' #name of the text file holding each layer's background threshold flux
    @property
    def LAYER_OFFSET_FILE_NAME_STEM(self) :
        return 'best_fit_offsets' #name of the .csv file holding each slide's LayerOffset result objects
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
    def BATCH_FF_DIRNAME_STEM(self) :
        return 'flatfield_BatchID'
    @property
    def INTENSITY_FIG_WIDTH(self) :
        return 16.8 #width of the intensity plot figure
    @property
    def ILLUMINATION_VARIATION_PLOT_WIDTH(self) :
        return 9.6 #width of the illumination variation plot
    #thresholding and masking
    @property
    def TISSUE_MASK_SMOOTHING_SIGMA(self) :
        return 5 #the sigma, in pixels, of the gaussian smoothing applied to images before thresholding/masking
    @property
    def BLUR_MASK_SMOOTHING_SIGMA(self) :
        return 1 #the sigma, in pixels, of the gaussian smoothing applied to images before calculating the laplacian variance for flagging blur
    @property
    def LOCAL_MEAN_KERNEL(self):
        return np.array([[0.0,0.2,0.0],
                         [0.2,0.2,0.2],
                         [0.0,0.2,0.0]]) #kernel to use for the local mean filter in getting the normalized laplacian variance for an image
    @property
    def WINDOW_EL(self) :
        if self._window_el is None :
            self._window_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45)) #window for computing the variance of the normalized laplacian
        return self._window_el
    @property
    def SMALLER_WINDOW_EL(self) :
        if self._smaller_window_el is None :
            self._smaller_window_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)) #window for taking the mean of the variance values in masking blur
        return self._smaller_window_el
    @property
    def TISSUE_MIN_SIZE(self) :
        return 2500 #minimum size in pixels of individual structure elements allowed in tissue masks
    @property
    def FOLD_MIN_PIXELS(self) :
        return 30000 #minimum number of pixels required to flag multilayer blur in images
    @property
    def FOLD_MIN_SIZE(self) :
        return 5000 #minimum size in pixels of individual structure elements allowed in multilayer blur masks
    @property
    def FOLD_NLV_CUT(self) :
        return 0.02 #0.025 #local normalized laplacian variance below which a pixel is flagged as blurry for multiple layer masks 
    @property
    def FOLD_MAX_MEAN(self) :
        return 0.018 #0.01875 #maximum mean within the smaller window of the local normalized laplacian variance allowed to flag multilayer blur
    @property
    def DUST_MIN_PIXELS(self) :
        return 30000 #minimum number of pixels required to flag DAPI layer blur in images
    @property
    def DUST_MIN_SIZE(self) :
        return 20000 #minimum size in pixels of individual structure elements allowed in DAPI layer blur masks
    @property
    def DUST_NLV_CUT(self) :
        return 0.005 #local normalized laplacian variance below which a pixel is flagged as blurry for multiple layer masks 
    @property
    def DUST_MAX_MEAN(self) :
        return 0.004 #maximum mean within the smaller window of the local normalized laplacian variance allowed to flag multilayer blur
    @property
    def SATURATION_MIN_PIXELS(self) :
        return 4500 #minimum number of pixels required to flag saturation in images
    @property
    def SATURATION_MIN_SIZE(self) :
        return 1000 #minimum size in pixels of individual structure elements allowed in saturation masks
    @property
    def SATURATION_INTENSITY_CUTS_35(self) :
        return [100,100,250,400,150] #intensity in counts/ms required to flag saturation in each layer group for 35-layer images
    @property
    def SATURATION_INTENSITY_CUTS_43(self) :
        return [100,-1,100,150,100,250,400] #intensity in counts/ms required to flag saturation in each layer group for 43-layer images
    #masking morphology transformations
    @property
    def SMALL_CO_EL(self) :
        if self._small_co_el is None :
            self._small_co_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)) #element for first small close/open in tissue masks
        return self._small_co_el
    @property
    def MEDIUM_CO_EL(self) :
        if self._medium_co_el is None :
            self._medium_co_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(18,18)) #element for medium-sized close/open morphology transformations
        return self._medium_co_el

    def __init__(self) :
        #some placeholders to only run functions once
        self._window_el = None
        self._smaller_window_el = None
        self._small_co_el = None
        self._medium_co_el = None

CONST=Const()
