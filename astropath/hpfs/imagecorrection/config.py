from ...utilities.config import CONST as UNIV_CONST

class ImageCorrectionConst :
    @property
    def DEFAULT_CORRECTION_MODEL_FILEPATH(self) :
        return UNIV_CONST.ASTROPATH_PROCESSING_DIR/'AstroPathCorrectionModels.csv'

IMAGECORRECTION_CONST = ImageCorrectionConst()