#imports 
import numpy as np

#class for shared constant variables
class Const :
    #file extensions
    @property
    def RAW_EXT(self) :
        return '.Data.dat' # extension of completely raw image files
    @property
    def IM3_EXT(self) :
        return '.im3' # extension of .im3 image files
    #image information
    @property
    def N_CLIP(self) :
        return 8 #number of pixels to clip from raw image edges
    @property
    def LAYER_GROUPS_35(self) :
        return [(1,9),(10,18),(19,25),(26,32),(33,35)] #first and last layer numbers in each broadband filter for 35-layer images
    @property
    def BRIGHTEST_LAYERS_35(self) :
        return [5,11,21,29,34] #brightest layers for 35-layer images (used in plotting)
    @property
    def DAPI_LAYER_GROUP_INDEX_35(self) :
        return 0 #index of the DAPI layer group for 35-layer images
    @property
    def RBC_LAYER_GROUP_INDEX_35(self) :
        return 1 #index of the layer group that tends to brightly show skin and red blood cells for 35-layer images 
    @property
    def LAYER_GROUPS_43(self) :
        return [(1,9),(10,11),(12,17),(18,20),(21,29),(30,36),(37,43)] #first and last layer numbers in each broadband filter for 43-layer images
    @property
    def BRIGHTEST_LAYERS_43(self) :
        return [5,10,16,19,22,31,41] #brightest layers for 43-layer images (used in plotting)
    @property
    def DAPI_LAYER_GROUP_INDEX_43(self) :
        return 0 #index of the DAPI layer group for 43-layer images
    @property
    def RBC_LAYER_GROUP_INDEX_43(self) :
        return 4 #index of the layer group that tends to brightly show skin and red blood cells for 43-layer images 
    #information for corrections
    @property
    def LAYER_ET_OFFSET_FILE_NAME_STEM(self) :
        return 'best_fit_offsets' #stem for the name of the .csv file holding a slide's LayerOffset result objects
    @property
    def BACKGROUND_THRESHOLD_TEXT_FILE_NAME_STEM(self) :
        return 'background_thresholds.txt' #name of the text file holding each layer's background threshold flux
    @property
    def FLATFIELD_IMAGE_DTYPE(self) :
        return np.float64 #datatype for the main output flatfield images
    @property
    def X_WARP_BIN_FILENAME(self) :
        return 'dx_warp_field' #name of the dx warping field binary file
    @property
    def Y_WARP_BIN_FILENAME(self) :
        return 'dy_warp_field' #name of the dy warping field binary file
    

CONST=Const()