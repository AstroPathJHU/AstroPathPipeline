from ..flatfield.config import CONST as FLATFIELD_CONST

#class for shared constant variables
class Const :
    #file extensions
    @property
    def RAW_EXT(self) :
        return '.Data.dat' # extension of completely raw image files
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
    def N_CLIP(self) :
        return 8 #number of pixels to clip from raw image edges
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
        return 'dx_warp_field.bin' #name of the dx warping field binary file
    @property
    def Y_WARP_BIN_FILENAME(self) :
        return 'dy_warp_field.bin' #name of the dy warping field binary file
    @property
    def WARP_FIELD_FIGURE_NAME(self) :
        return 'warp_fields.png'
    @property
    def OVERLAY_NORMALIZE(self) :
        return 1000. #value to use to normalize the overlay images
    @property 
    def OCTET_OVERLAP_COMPARISON_FIGURE_WIDTH(self) :
        return 3*6.4 #width of the octet overlap comparison figures
    #setup
    @property
    def MICROSCOPE_OBJECTIVE_FOCAL_LENGTH(self) :
        return 40000. #focal length of the microscope objective (20mm) in pixels
    
CONST=Const()
