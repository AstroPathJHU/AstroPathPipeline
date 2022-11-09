#imports
import random
from ...utilities.config import CONST as UNIV_CONST
from ...shared.samplemetadata import MetadataSummary
from ...shared.rectangle import RectangleReadComponentTiffMultiLayer, RectangleCorrectedIm3MultiLayer
from ...shared.overlap import Overlap
from ...shared.sample import WorkflowSample, XMLLayoutReaderTissue
from .meanimagesample import MeanImageSampleBase, MeanImageSampleBaseComponentTiff, MeanImageSampleBaseIm3

class AppliedFlatfieldSampleBase(MeanImageSampleBase,WorkflowSample,XMLLayoutReaderTissue) :
    """
    Class to use in running most of the MeanImageSample functions but handling the output differently
    """

    multilayer = True
    overlaptype = Overlap
    nclip = UNIV_CONST.N_CLIP

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)
        self._metadata_summary_ff = None
        self._metadata_summary_cmi = None

    def run(self) :
        """
        Not implemented in the base class
        """
        pass

    @property
    def flatfield_rectangles(self) :
        return self._flatfield_rectangles
    @property
    def meanimage_rectangles(self) :
        return self._meanimage_rectangles
    @property
    def flatfield_metadata_summary(self) :
        return self._metadata_summary_ff
    @property
    def corrected_meanimage_metadata_summary(self) :
        return self._metadata_summary_cmi

    @classmethod
    def getoutputfiles(cls,**kwargs) :
        return [*super().getoutputfiles(**kwargs)]
    @classmethod
    def workflowdependencyclasses(cls, **kwargs):
        return super().workflowdependencyclasses(**kwargs)

class AppliedFlatfieldSampleComponentTiff(AppliedFlatfieldSampleBase,MeanImageSampleBaseComponentTiff) :
    """
    For applying flatfields to orthogonal subsets of component tiff data
    """

    rectangletype = RectangleReadComponentTiffMultiLayer

    def __init__(self,*args,image_set_split='random',**kwargs) :
        super().__init__(*args,**kwargs)
        if len(self.tissue_bulk_rects)>1 :
            if image_set_split=='random' :
                self._flatfield_rectangles = random.sample(self.tissue_bulk_rects,round(len(self.tissue_bulk_rects)/2))
            elif image_set_split=='sequential' :
                self._flatfield_rectangles = self.tissue_bulk_rects[:round(len(self.tissue_bulk_rects)/2)]
            self._meanimage_rectangles = [r for r in self.tissue_bulk_rects if r not in self._flatfield_rectangles]
        else :
            self._flatfield_rectangles = []
            self._meanimage_rectangles = []

    def run(self) :
        #find the masks for the sample if they're needed
        if not self.skip_masking :
            self.set_image_masking_dirpath()
            if not self.use_precomputed_masks :
                raise NotImplementedError('ERROR: cannot run MeanImageSampleComponentTiff without pre-computed masks!')
        #set the metadata summaries for the sample
        ff_rect_ts = [r.t for r in self._flatfield_rectangles]
        cmi_rect_ts = [r.t for r in self._meanimage_rectangles]
        self._metadata_summary_ff = MetadataSummary(self.SlideID,self.Project,self.Cohort,self.microscopename,
                                                     str(min(ff_rect_ts)),str(max(ff_rect_ts)))
        self._metadata_summary_cmi = MetadataSummary(self.SlideID,self.Project,self.Cohort,self.microscopename,
                                                      str(min(cmi_rect_ts)),str(max(cmi_rect_ts)))

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.componenttifffile for r in self.rectangles),
               ]

    @classmethod
    def logmodule(cls) : 
        return "appliedflatfield_comp_tiff"

class AppliedFlatfieldSampleIm3(AppliedFlatfieldSampleBase,MeanImageSampleBaseIm3) :
    """
    For applying flatfields to orthogonal subsets of im3 data
    """

    rectangletype = RectangleCorrectedIm3MultiLayer

    def __init__(self,*args,image_set_split='random',**kwargs) :
        super().__init__(*args,**kwargs)
        if len(self.tissue_bulk_rects)>1 :
            if image_set_split=='random' :
                self._flatfield_rectangles = random.sample(self.tissue_bulk_rects,round(len(self.tissue_bulk_rects)/2))
            elif image_set_split=='sequential' :
                self._flatfield_rectangles = self.tissue_bulk_rects[:round(len(self.tissue_bulk_rects)/2)]
            self._meanimage_rectangles = [r for r in self.tissue_bulk_rects if r not in self._flatfield_rectangles]
        else :
            self._flatfield_rectangles = []
            self._meanimage_rectangles = []

    def run(self) :
        #find or create the masks for the sample if they're needed
        if not self.skip_masking :
            self.create_or_find_image_masks()
        #set the metadata summaries for the sample
        ff_rect_ts = [r.t for r in self._flatfield_rectangles]
        cmi_rect_ts = [r.t for r in self._meanimage_rectangles]
        self._metadata_summary_ff = MetadataSummary(self.SlideID,self.Project,self.Cohort,self.microscopename,
                                                     str(min(ff_rect_ts)),str(max(ff_rect_ts)))
        self._metadata_summary_cmi = MetadataSummary(self.SlideID,self.Project,self.Cohort,self.microscopename,
                                                      str(min(cmi_rect_ts)),str(max(cmi_rect_ts)))

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.im3file for r in self.rectangles),
               ]

    @classmethod
    def logmodule(cls) : 
        return "appliedflatfield"