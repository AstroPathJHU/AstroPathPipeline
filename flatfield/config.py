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
    def FILE_EXT(self) :
        return '.bin' #file extension for the main output files
    @property
    def FLATFIELD_FILE_NAME_STEM(self) :
        return 'flatfield' #what the flatfield file is called
    @property
    def SMOOTHED_CORRECTED_MEAN_IMAGE_FILE_NAME_STEM(self) :
        return 'smoothed_corrected_mean_image' #name of the outputted smoothed corrected mean image file
    @property
    def THRESHOLD_TEXT_FILE_NAME_STEM(self) :
        return 'background_thresholds.txt' #name of the text file holding each layer's background threshold flux
    @property
    def LAYER_OFFSET_FILE_NAME_STEM(self) :
        return 'best_fit_offsets' #name of the .csv file holding each sample's LayerOffset result objects
    #image smoothing
    @property
    def GENTLE_GAUSSIAN_SMOOTHING_SIGMA(self) :
        return 5 #the sigma, in pixels, of the gentle gaussian smoothing applied to images before thresholding/masking
    #masking
    @property 
    def MASKING_PLOT_FIG_SIZE(self) :
        return (12.8,18.4) #size of the outputted masking plot
    #masking morphology transformations
    @property
    def CO1_EL(self) :
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)) #element for first close/open (and final, repeated, open)
    @property
    def CO2_EL(self) :
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(16,16)) #element for second close/open
    @property
    def C3_EL(self) :
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45)) #element for large-scale close
    @property
    def OPEN_3_ITERATIONS(self) :
        return 3 #number of iterations for final small-scale open
    #image information
    @property
    def N_CLIP(self) :
        return 8 #number of pixels to clip from raw image edges

CONST=Const()
