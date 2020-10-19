#imports
from ..flatfield.config import CONST as FLATFIELD_CONST

#class for shared constant variables
class Const :
    #file extensions
    @property
    def RAW_EXT(self) :
        return FLATFIELD_CONST.RAW_EXT # extension of completely raw image files
    @property
    def IM3_EXT(self) :
        return FLATFIELD_CONST.IM3_EXT # extension of .im3 image files
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
    #overlap cost parameterization
    @property
    def OVERLAP_COST_PARAMETERIZATION_N_POINTS(self) :
        return 100 #number of points to test between the bounds for each overlap's cost parameterization
    
CONST=Const()