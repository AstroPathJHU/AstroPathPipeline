#imports
import pathlib
import numpy as np

class SegmentationConst :
    @property
    def NNUNET_MODEL_TOP_DIR(self) :
        return pathlib.Path(__file__).parent/'nnunet_models' #directory holding nnUnet model results to use
    @property
    def IHC_PCA_BLACK_COMPONENT(self) :
        #first PCA vector component corresponding to the black stain in the IHC images
        return np.array([0.5488114581299663,0.5860882031075596,0.5960760032105071])
    @property
    def IHC_MEMBRANE_LAYER_NORM(self) :
        #normalization for the membrane layer from the PCA transform of the IHC image
        return 15.

SEG_CONST = SegmentationConst()
