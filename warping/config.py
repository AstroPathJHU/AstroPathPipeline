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
    #how to handle images
    @property
    def smoothsigma(self):
        return 1.0 #default sigma for smoothing images loaded for warping
    #files that get created
    @property 
    def OCTET_OVERLAP_CSV_FILE_NAMESTEM(self) :
        return '_overlap_octets.csv' #stem for the name of the octet overlap csv file name
    #setup
    @property
    def MICROSCOPE_OBJECTIVE_FOCAL_LENGTH(self) :
        return 40000. #focal length of the microscope objective (20mm) in pixels
    
CONST=Const()
