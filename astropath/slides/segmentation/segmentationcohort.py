#imports
from ...shared.argumentparser import SegmentationFolderArgumentParser
from ...shared.cohort import ParallelCohort, WorkflowCohort
from .segmentationsample import SegmentationSampleBase, SegmentationSampleNNUNet
from .segmentationsample import SegmentationSampleDeepCell, SegmentationSampleMesmer

class SegmentationCohortBase(ParallelCohort,WorkflowCohort,SegmentationFolderArgumentParser) :
    sampleclass = SegmentationSampleBase
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,segmentationfolder=None,**kwargs) :
        super().__init__(*args,**kwargs)
        self.segmentationfolder = segmentationfolder

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
            'segmentationfolder':self.segmentationfolder,
            }

    @property
    def workflowkwargs(self) :
        return {**super().workflowkwargs,
            'segmentationfolder':self.segmentationfolder,
        }

class SegmentationCohortNNUNet(SegmentationCohortBase) :
    sampleclass = SegmentationSampleNNUNet

class SegmentationCohortDeepCell(SegmentationCohortBase) :
    sampleclass = SegmentationSampleDeepCell

class SegmentationCohortMesmer(SegmentationCohortBase) :
    sampleclass = SegmentationSampleMesmer

def segmentationcohortnnunet(args=None) :
    SegmentationCohortNNUNet.runfromargumentparser(args)

def segmentationcohortdeepcell(args=None) :
    SegmentationCohortDeepCell.runfromargumentparser(args)

def segmentationcohortmesmer(args=None) :
    SegmentationCohortMesmer.runfromargumentparser(args)
