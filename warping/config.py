#imports
from ..flatfield.config import CONST as FLATFIELD_CONST
from ..exposuretime.config import CONST as ET_CONST
import numpy as np

#class for shared constant variables
class Const :
    #some default command line arguments
    @property
    def DEFAULT_FIXED(self) :
        return 'fx,fy'
    @property
    def DEFAULT_NORMALIZE(self) :
        return 'cx,cy,fx,fy,k1,k2,k3,p1,p2'
    @property
    def DEFAULT_INIT_PARS(self) :
        return {'cx':None,'cy':None,'fx':None,'fy':None,'k1':None,'k2':None,'k3':None,'p1':None,'p2':None}
    @property
    def DEFAULT_INIT_BOUNDS(self) :
        return {'cx':None,'cy':None,'fx':None,'fy':None,'k1':None,'k2':None,'k3':None,'p1':None,'p2':None}
    #file extensions
    @property
    def IM3_EXT(self) :
        return ET_CONST.IM3_EXT # extension of im3 files
    @property
    def RAW_EXT(self) :
        return ET_CONST.RAW_EXT # extension of completely raw image files
    @property
    def WARP_EXT(self) :
        return '.camWarp_layer' #extension stem for the warped files
    @property
    def FW_EXT(self) :
        return '.fw' #extension stem for the flatfielded/warped files
    @property
    def THRESHOLD_FILE_EXT(self) :
        return f'_{FLATFIELD_CONST.THRESHOLD_TEXT_FILE_NAME_STEM}'
    #other image properties
    @property
    def CORNER_OVERLAP_TAGS(self):
        return [1,3,7,9] #list of tags representing overlaps that are corners 
    @property
    def FLATFIELD_DTYPE(self):
        return FLATFIELD_CONST.IMG_DTYPE_OUT #datatype of the flatfield image
    @property
    def N_CLIP(self) :
        return FLATFIELD_CONST.N_CLIP #number of pixels to clip from raw image edges
    #how to handle images
    @property
    def SMOOTH_SIGMA(self):
        return 1.0 #default sigma for smoothing images loaded for warping
    #files that get created
    @property 
    def OCTET_OVERLAP_CSV_FILE_NAMESTEM(self) :
        return '_overlap_octets.csv' #stem for the name of the octet overlap csv file name
    @property
    def FIT_RESULT_CSV_FILE_NAME(self) :
        return 'fit_result.csv' #the name of the fit result text file that gets written out
    @property
    def X_WARP_BIN_FILENAME(self) :
        return 'dx_warp_field' #name of the dx warping field binary file
    @property
    def Y_WARP_BIN_FILENAME(self) :
        return 'dy_warp_field' #name of the dy warping field binary file
    @property
    def OUTPUT_FIELD_DTYPE(self) :
        return np.float64 #datatype for outputting the warp fields
    @property
    def WARP_FIELD_FIGURE_NAME(self) :
        return 'warp_fields'
    @property
    def OVERLAY_NORMALIZE(self) :
        return 1000. #value to use to normalize the overlay images
    @property 
    def OCTET_OVERLAP_COMPARISON_FIGURE_WIDTH(self) :
        return 2*6.4 #width of the octet overlap comparison figures
    #setup
    @property
    def MICROSCOPE_OBJECTIVE_FOCAL_LENGTH(self) :
        return 40000. #focal length of the microscope objective (20mm) in pixels
    
CONST=Const()
