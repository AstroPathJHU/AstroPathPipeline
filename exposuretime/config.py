#imports
from ..flatfield.config import CONST as FLATFIELD_CONST

#class for shared constant variables
class Const :
    #file extensions
    @property
    def RAW_EXT(self) :
        return '.Data.dat' # extension of completely raw image files
    @property
    def IM3_EXT(self) :
        return '.im3' # extension of .im3 image files
    #filenames
    @property
    def LAYER_OFFSET_FILE_NAME_STEM(self) :
    	return FLATFIELD_CONST.LAYER_OFFSET_FILE_NAME_STEM #name of the layer offset result .csv file
    #image properties
    @property
    def FLATFIELD_DTYPE(self):
        return FLATFIELD_CONST.IMG_DTYPE_OUT #datatype of the flatfield image
    @property
    def N_CLIP(self) :
        return FLATFIELD_CONST.N_CLIP #number of pixels to clip from raw image edges
    
CONST=Const()