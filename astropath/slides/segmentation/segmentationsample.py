#imports
import methodtools
from ...shared.sample import ParallelSample, ReadRectanglesComponentAndIHCTiffFromXML
from ...shared.sample import SampleWithSegmentationFolder, WorkflowSample
from .config import SEG_CONST

class SegmentationSampleBase(ReadRectanglesComponentAndIHCTiffFromXML,SampleWithSegmentationFolder,
                             WorkflowSample,ParallelSample) :
    """
    Write out nuclear segmentation maps based on the DAPI layers of component tiffs for a single sample
    Algorithms available include pre-trained nnU-Net and DeepCell/mesmer models 
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,layercomponenttiff=1,**kwargs) :
        # only need to load the DAPI layers of the rectangles, so send that to the __init__
        if layercomponenttiff != 1 :
            raise RuntimeError(f'ERROR: sample layer was set to {kwargs.get("layer")}')
        super().__init__(*args,layercomponenttiff=layercomponenttiff,**kwargs)

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.componenttifffile for r in self.rectangles),
               ]

    def run(self,**kwargs) :
        if not self.segmentationfolder.is_dir() :
            self.segmentationfolder.mkdir(parents=True)
        self.runsegmentation(**kwargs)

    @methodtools.lru_cache()
    def runsegmentation(self, **kwargs): pass

    #################### CLASS METHODS ####################

    @classmethod
    def getoutputfiles(cls,SlideID,segmentationroot,informdataroot,segmentationfolderarg,**otherworkflowkwargs) :
        outputdir=cls.segmentation_folder(segmentationfolderarg,segmentationroot,SlideID)
        append = None
        if cls.segmentationalgorithm()=='nnunet' :
            append = SEG_CONST.NNUNET_SEGMENT_FILE_APPEND
        elif cls.segmentationalgorithm()=='deepcell' :
            append = SEG_CONST.DEEPCELL_SEGMENT_FILE_APPEND
        elif cls.segmentationalgorithm()=='mesmer' :
            append = SEG_CONST.MESMER_SEGMENT_FILE_APPEND
        file_stems = [fp.name[:-len('_component_data.tif')] 
                      for fp in (informdataroot/SlideID/'inform_data'/'Component_Tiffs').glob('*_component_data.tif')]
        outputfiles = []
        for stem in file_stems :
            outputfiles.append(outputdir/f'{stem}_{append}')
        return outputfiles

    @classmethod
    def workflowdependencyclasses(cls, **kwargs):
        return super().workflowdependencyclasses(**kwargs)
