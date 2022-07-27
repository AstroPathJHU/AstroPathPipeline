#imports
import pathlib
import numpy as np

class SegmentationConst :
    @property
    def NNUNET_MODEL_TOP_DIR(self) :
        return pathlib.Path(__file__).parent/'nnunet_models' #directory holding nnUnet model results to use
    @property
    def NNUNET_SEGMENT_FILE_APPEND(self) :
        #append for nnunet segmentation files
        return 'nnunet_nuclear_segmentation.npz' 
    @property
    def GROUP_SIZE(self) :
        #image group size to use when running DeepCell/Mesmer
        return 48
    @property
    def IHC_PCA_BLACK_COMPONENT(self) :
        #first PCA vector component corresponding to the black stain in the IHC images
        return np.array([0.5488114581299663,0.5860882031075596,0.5960760032105071])
    @property
    def IHC_MEMBRANE_LAYER_NORM(self) :
        #normalization for the membrane layer from the PCA transform of the IHC image
        return 15.
    @property
    def DEEPCELL_SEGMENT_FILE_APPEND(self) :
        #append for deepcell nuclear segmentation files
        return 'deepcell_nuclear_segmentation.npz'
    @property
    def MESMER_SEGMENT_FILE_APPEND(self) :
        #append for mesmer whole-cell + nuclear segmentation files
        return 'mesmer_segmentation.npz'
    @property
    def MEMBRANE_LAYER_TARGETS(self) :
        #list of names of targets in MergeConfig files corresponding to pan-membrane stains
        target_strings = ['ATPase/CD44/CD45']
        return [ts.replace('/','').lower() for ts in target_strings]

SEG_CONST = SegmentationConst()
