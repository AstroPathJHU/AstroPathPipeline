#imports
from ...utilities.config import CONST as UNIV_CONST
from ...shared.sample import TissueSampleBase, WorkflowSample
from .config import CONST
from .meanimagesample import MeanImageSampleBase, MeanImageSampleComponentTiffTissue, MeanImageSampleIm3Tissue

class BatchFlatfieldSampleBase(WorkflowSample) :
    """
    Small utility class to hold sample-dependent information for batch flatfield runs
    Just requires as input files the relevant output of the meanimage mode
    """

    multilayer = True

    def __init__(self,*args,version,meanimage_dirname,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__meanimage_dirname = meanimage_dirname
        self.__version = version

    @property
    def meanimagefolder(self) :
        return self.im3folder/self.__meanimage_dirname
    @property
    def meanimage(self) :
        return self.meanimagefolder/f'{self.SlideID}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}'
    @property
    def sumimagessquared(self) :
        return self.meanimagefolder/f'{self.SlideID}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}'
    @property
    def maskstack(self) :
        return self.meanimagefolder/f'{self.SlideID}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}'
    @property
    def fieldsused(self) :
        return self.meanimagefolder/CONST.FIELDS_USED_CSV_FILENAME
    @property
    def metadatasummary(self) :
        return self.meanimagefolder/f'{self.SlideID}-{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}'
    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                self.meanimage,self.sumimagessquared,self.maskstack,self.fieldsused,self.metadatasummary]
    def run(self,flatfield,samplesprocessed,totalsamples) :
        msg = f'Adding mean image and mask stack from {self.SlideID} meanimage directory "{self.meanimagefolder}" '
        msg+= f'to flatfield model version {self.__version} ({len(samplesprocessed)+1} of {totalsamples})....'
        self.logger.info(msg)
        flatfield.add_batchflatfieldsample(self)
        samplesprocessed.append(self)
    @classmethod
    def getoutputfiles(cls,**kwargs) :
        return [*super().getoutputfiles(**kwargs)]
    @classmethod
    def defaultunits(cls) :
        return MeanImageSampleBase.defaultunits()

class BatchFlatfieldSampleComponentTiff(BatchFlatfieldSampleBase) :
    """
    A class for making batch flatfields from the output of component tiff meanimage runs
    """

    def __init__(self,*args,meanimage_dirname=f'{UNIV_CONST.MEANIMAGE_DIRNAME}_comp_tiff',**kwargs) :
        super().__init__(*args,meanimage_dirname=meanimage_dirname,**kwargs)

    @classmethod
    def logmodule(cls) : 
        return "batchflatfield_comp_tiff"

class BatchFlatfieldSampleIm3(BatchFlatfieldSampleBase) :
    """
    A class for making batch flatfields from the output of im3 meanimage runs
    """

    def __init__(self,*args,meanimage_dirname=UNIV_CONST.MEANIMAGE_DIRNAME,**kwargs) :
        super().__init__(*args,meanimage_dirname=meanimage_dirname,**kwargs)

    @classmethod
    def logmodule(cls) : 
        return "batchflatfield"

class BatchFlatfieldSampleIm3Tissue(BatchFlatfieldSampleIm3, TissueSampleBase) :
    @classmethod
    def workflowdependencyclasses(cls, **kwargs):
        return [*super().workflowdependencyclasses(**kwargs),MeanImageSampleIm3Tissue]

class BatchFlatfieldSampleComponentTiffTissue(BatchFlatfieldSampleComponentTiff, TissueSampleBase) :
    @classmethod
    def workflowdependencyclasses(cls, **kwargs):
        return [*super().workflowdependencyclasses(**kwargs),MeanImageSampleComponentTiffTissue]
