#imports
import cv2
import numpy as np

#class for shared constant variables
class Const :
    @property
    def TISSUE_MASK_SMOOTHING_SIGMA(self) :
        return 5 #the sigma, in pixels, of the gaussian smoothing applied to images before thresholding/masking
    @property
    def BLUR_MASK_SMOOTHING_SIGMA(self) :
        """
        the sigma, in pixels, of the gaussian smoothing applied to (Vectra 3.0) images before 
        calculating the laplacian variance for flagging blur
        """
        return 1 
    @property
    def LOCAL_MEAN_KERNEL(self):
        """
        kernel to use for the local mean filter in getting the normalized laplacian variance for an image
        """
        return np.array([[0.0,0.2,0.0],
                         [0.2,0.2,0.2],
                         [0.0,0.2,0.0]])
    @property
    def WINDOW_EL(self) :
        """
        window for computing the variance of the normalized laplacian
        """
        if self.__window_el is None :
            self.__window_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45))
        return self.__window_el
    @property
    def SMALLER_WINDOW_EL(self) :
        """
        window for taking the mean of the variance values in masking blur
        """
        if self.__smaller_window_el is None :
            self.__smaller_window_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        return self.__smaller_window_el
    @property
    def TISSUE_MIN_SIZE(self) :
        return 2500 #minimum size in pixels of individual structure elements allowed in tissue masks
    @property
    def FOLD_MIN_PIXELS(self) :
        return 30000 #minimum number of pixels required to flag multilayer blur in images
    @property
    def FOLD_MIN_SIZE(self) :
        return 3000 #minimum size in pixels of individual structure elements allowed in multilayer blur masks
    @property
    def FOLD_NLV_CUTS(self) :
        """
        local normalized laplacian variances below which a pixel is flagged as blurry for multiple layer masks 
        """
        return {'vectra_dapi':0.0035,
                'vectra_fitc':0.0035,
                'vectra_cy3':0.0035,
                'vectra_texasred':0.0035,
                'vectra_cy5':0.0035,
                'polaris_dapi':0.120,
                'polaris_opal780':0.0000,
                'polaris_opal480':0.0700,
                'polaris_cy5':0.0700,
                'polaris_fitc':0.1200,
                'polaris_cy3':0.1000,
                'polaris_texasred':0.1000,
            }
    @property
    def FOLD_MAX_MEANS(self) :
        """
        maximum mean within the smaller window of the local normalized laplacian variance 
        allowed to flag multilayer blur
        """
        return {'vectra_dapi':0.0030,
                'vectra_fitc':0.0030,
                'vectra_cy3':0.0030,
                'vectra_texasred':0.0030,
                'vectra_cy5':0.0030,
                'polaris_dapi':0.100,
                'polaris_opal780':0.000,
                'polaris_opal480':0.060,
                'polaris_cy5':0.060,
                'polaris_fitc':0.100,
                'polaris_cy3':0.085,
                'polaris_texasred':0.085,
            }
    @property
    def DUST_MIN_PIXELS(self) :
        return 30000 #minimum number of pixels required to flag DAPI layer blur in images
    @property
    def DUST_MIN_SIZE(self) :
        return 20000 #minimum size in pixels of individual structure elements allowed in DAPI layer blur masks
    @property
    def DUST_NLV_CUTS(self) :
        """
        local normalized laplacian variance below which a pixel is flagged as blurry for multiple layer masks 
        """
        return {'vectra_dapi':0.00085,
                'polaris_dapi':0.025,
            }
    @property
    def DUST_MAX_MEANS(self) :
        """
        maximum mean within the smaller window of the local normalized laplacian variance 
        allowed to flag multilayer blur
        """
        return {'vectra_dapi':0.00065,
                'polaris_dapi':0.01875,
            }
    @property
    def BLUR_FLAG_STRING(self) :
        return 'blurred likely folded tissue or dust' #string to use for blurred areas in the labelled mask regions file
    @property
    def SATURATION_MIN_PIXELS(self) :
        return 4500 #minimum number of pixels required to flag saturation in images
    @property
    def SATURATION_MIN_SIZE(self) :
        return 1000 #minimum size in pixels of individual structure elements allowed in saturation masks
    @property
    def SATURATION_INTENSITY_CUTS(self) :
        """
        intensity in counts/ms required to flag saturation in each layer group
        """
        return {'vectra_dapi':100,
                'vectra_fitc':100,
                'vectra_cy3':250,
                'vectra_texasred':400,
                'vectra_cy5':150,
                'polaris_dapi':100,
                'polaris_opal780':50,
                'polaris_opal480':250,
                'polaris_cy5':250,
                'polaris_fitc':150,
                'polaris_cy3':100,
                'polaris_texasred':100,
            }
    @property
    def SATURATION_FLAG_STRING(self) :
        """
        descriptive string to use for saturated areas in the labelled mask regions file
        """
        return 'saturated likely skin or red blood cells or stain'
    #masking morphology transformations
    @property
    def SMALL_CO_EL(self) :
        """
        element for first small close/open in tissue masks
        """
        if self.__small_co_el is None :
            self.__small_co_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        return self.__small_co_el
    @property
    def MEDIUM_CO_EL(self) :
        """
        element for medium-sized close/open morphology transformations
        """
        if self.__medium_co_el is None :
            self.__medium_co_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(18,18))
        return self.__medium_co_el

    def __init__(self) :
        #some placeholders to only run functions once
        self.__window_el = None
        self.__smaller_window_el = None
        self.__small_co_el = None
        self.__medium_co_el = None

CONST=Const()