#imports
import pathlib

class SegmentationConst :
    @property
    def NNUNET_MODEL_TOP_DIR(self) :
        return pathlib.Path(__file__).parent/'nnunet_models' #directory holding nnUnet model results to use
    @property
    def SEGMENTATION_DIR_NAME(self) :
        return 'segmentation' #name of the directory holding segmentation output(s) for each sample

SEG_CONST = SegmentationConst()