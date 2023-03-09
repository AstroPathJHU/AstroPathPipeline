#imports
import pathlib
import numpy as np

class SegmentationConst :
    @property
    def NNUNET_MODEL_TOP_DIR(self) :
        """
        Top of directory holding nnUNet model results to use
        """
        return (pathlib.Path(__file__).parent/'nnunet_models').resolve()
    @property
    def NNUNET_TASK_NAME(self) :
        return 'Task500_Pathology_DAPI'
    @property
    def NNUNET_MODEL_DIR(self) :
        """
        Path to the directory that holds the pre-trained nnUNet model files
        """
        return self.NNUNET_MODEL_TOP_DIR/'nnUNet'/'2d'/self.NNUNET_TASK_NAME/'nnUNetTrainerV2__nnUNetPlansv2.1'
    @property
    def NNUNET_MODEL_FILES(self) :
        """
        A list of all of the pre-trained nnUNet model files
        """
        model_files = [
            self.NNUNET_MODEL_DIR/'plans.pkl',
            self.NNUNET_MODEL_DIR/'postprocessing.json',
        ]
        for fold_n in (0,1,2,3,4) :
            model_files+=[
                self.NNUNET_MODEL_DIR/f'fold_{fold_n}'/'model_final_checkpoint.model.pkl',
                self.NNUNET_MODEL_DIR/f'fold_{fold_n}'/'model_final_checkpoint.model',
            ]
        return model_files
    @property
    def NNUNET_MODEL_FILES_URL(self) :
        return 'https://data.idies.jhu.edu/bki-nnunet/' #URL where the NNUNet model files are stored
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
