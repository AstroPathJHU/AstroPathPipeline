#imports
import pathlib

#class for shared constant variables
class Const :
    #file extensions
    @property
    def RAW_EXT(self) :
        return '.Data.dat' # extension of completely raw image files shredded from .im3 files
    @property
    def FLATW_EXT(self) :
        return '.fw' # extension of corrected "flatw" image files
    @property
    def IM3_EXT(self) :
        return '.im3' # extension of .im3 image files
    @property
    def QPTIFF_SUFFIX(self) :
        return '_qptiff.jpg' # suffix for the qptiff files
    @property
    def EXPOSURE_XML_EXTS(self) :
        return ['.SpectralBasisInfo.Exposure.xml',
                '.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml'] #extensions for exposure time xml files
    #subdirectory names
    @property
    def IM3_DIR_NAME(self) :
        return 'im3' #name of the im3 subdirectory
    @property
    def DBLOAD_DIR_NAME(self) :
        return 'dbload' #name of the dbload directory
    @property
    def MEANIMAGE_DIRNAME(self) :
        return 'meanimage' #name of the output directory for slidemeanimage results
    @property
    def FLATFIELD_DIRNAME(self) :
        return 'flatfield' #name of the directory holding flatfield model results for an entire cohort (directory with this name gets created in root)
    @property
    def WARPING_DIRNAME(self) :
        return 'warping' #name of the directory holding warping model results for an entire cohort (directory with this name gets created in root)
    @property
    def ASTROPATH_PROCESSING_DIR(self) :
        return pathlib.Path('//bki04/astropath_processing')
    #image information
    @property
    def N_CLIP(self) :
        return 8 #number of pixels to clip from raw image edges
    @property
    def COMP_TIFF_DAPI_LAYER(self) :
        return 1 #number of the DAPI layer in component tiff images
    @property
    def COMP_TIFF_AF_LAYER(self) :
        return 8 #number of the autofluorescence (AF) layer in component tiff images

CONST=Const()