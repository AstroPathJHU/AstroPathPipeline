#imports
import numpy as np
import cv2

#class for shared constant variables
class Const :
    @property
    def TISSUE_MASK_SMOOTHING_SIGMA(self) :
        return 5 #the sigma, in pixels, of the gaussian smoothing applied to images before thresholding/masking
    @property
    def BLUR_MASK_SMOOTHING_SIGMA(self) :
        return 1 #the sigma, in pixels, of the gaussian smoothing applied to (Vectra 3.0) images before calculating the laplacian variance for flagging blur
    @property
    def LOCAL_MEAN_KERNEL(self):
        return np.array([[0.0,0.2,0.0],
                         [0.2,0.2,0.2],
                         [0.0,0.2,0.0]]) #kernel to use for the local mean filter in getting the normalized laplacian variance for an image
    @property
    def WINDOW_EL(self) :
        if self.__window_el is None :
            self.__window_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45)) #window for computing the variance of the normalized laplacian
        return self.__window_el
    @property
    def SMALLER_WINDOW_EL(self) :
        if self.__smaller_window_el is None :
            self.__smaller_window_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)) #window for taking the mean of the variance values in masking blur
        return self.__smaller_window_el
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
    def FOLD_NLV_CUTS_BY_LAYER_GROUP_35(self) :
        return [0.0035,0.0035,0.0035,0.0035,0.0035] #local normalized laplacian variances below which a pixel is flagged as blurry for multiple layer masks 
    @property
    def FOLD_MAX_MEANS_BY_LAYER_GROUP_35(self) :
        return [0.0030,0.0030,0.0030,0.0030,0.0030] #maximum mean within the smaller window of the local normalized laplacian variance allowed to flag 
                                                    #multilayer blur
    @property
    def FOLD_NLV_CUTS_BY_LAYER_GROUP_43(self) :
        return [0.120,0.0000,0.0700,0.0700,0.1200,0.1000,0.1000] 
    @property
    def FOLD_MAX_MEANS_BY_LAYER_GROUP_43(self) :
        return [0.100,0.0000,0.0600,0.0600,0.1000,0.0850,0.0850] 
    @property
    def DUST_MIN_PIXELS(self) :
        return 30000 #minimum number of pixels required to flag DAPI layer blur in images
    @property
    def DUST_MIN_SIZE(self) :
        return 20000 #minimum size in pixels of individual structure elements allowed in DAPI layer blur masks
    @property
    def DUST_NLV_CUT_35(self) :
        return 0.00085 #local normalized laplacian variance below which a pixel is flagged as blurry for multiple layer masks 
    @property
    def DUST_MAX_MEAN_35(self) :
        return 0.00065 #maximum mean within the smaller window of the local normalized laplacian variance allowed to flag multilayer blur
    @property
    def DUST_NLV_CUT_43(self) :
        return 0.0700
    @property
    def DUST_MAX_MEAN_43(self) :
        return 0.0525
    @property
    def BLUR_FLAG_STRING(self) :
    	return 'blurred likely folded tissue or dust' #descriptive string to use for blurred areas in the labelled mask regions file
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
        return [100,50,250,250,150,100,50] #intensity in counts/ms required to flag saturation in each layer group for 43-layer images
    @property
    def SATURATION_FLAG_STRING(self) :
        return 'saturated likely skin or red blood cells or stain' #descriptive string to use for saturated areas in the labelled mask regions file
    #masking morphology transformations
    @property
    def SMALL_CO_EL(self) :
        if self.__small_co_el is None :
            self.__small_co_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)) #element for first small close/open in tissue masks
        return self.__small_co_el
    @property
    def MEDIUM_CO_EL(self) :
        if self.__medium_co_el is None :
            self.__medium_co_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(18,18)) #element for medium-sized close/open morphology transformations
        return self.__medium_co_el

    def __init__(self) :
        #some placeholders to only run functions once
        self.__window_el = None
        self.__smaller_window_el = None
        self.__small_co_el = None
        self.__medium_co_el = None

CONST=Const()