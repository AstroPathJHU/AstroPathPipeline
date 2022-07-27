#imports
from ...shared.cohort import ParallelCohort, SegmentationFolderCohort, SelectRectanglesCohort, WorkflowCohort
from .segmentationsample import SegmentationSampleBase
from .segmentationsamplennunet import SegmentationSampleNNUNet
from .segmentationsampledeepcell import SegmentationSampleDeepCell
from .segmentationsamplemesmer import SegmentationSampleMesmer

class SegmentationCohortBase(ParallelCohort,SelectRectanglesCohort,WorkflowCohort,SegmentationFolderCohort) :
    sampleclass = SegmentationSampleBase
    __doc__ = sampleclass.__doc__

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
