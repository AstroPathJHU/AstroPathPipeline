#imports
from ...shared.cohort import ParallelCohort, SegmentationFolderCohort, SelectRectanglesCohort, WorkflowCohort
from .segmentationsample import SegmentationSampleBase
from .segmentationsamplennunet import SegmentationSampleNNUNet
from .segmentationsampledeepcell import SegmentationSampleDeepCell
from .segmentationsamplemesmer import SegmentationSampleMesmerWithIHC, SegmentationSampleMesmerComponentTiff

class SegmentationCohortBase(ParallelCohort,SelectRectanglesCohort,WorkflowCohort,SegmentationFolderCohort) :
    sampleclass = SegmentationSampleBase
    __doc__ = sampleclass.__doc__

class SegmentationCohortNNUNet(SegmentationCohortBase) :
    sampleclass = SegmentationSampleNNUNet

class SegmentationCohortDeepCell(SegmentationCohortBase) :
    sampleclass = SegmentationSampleDeepCell

class SegmentationCohortMesmerWithIHC(SegmentationCohortBase) :
    sampleclass = SegmentationSampleMesmerWithIHC

class SegmentationCohortMesmerComponentTiff(SegmentationCohortBase) :
    sampleclass = SegmentationSampleMesmerComponentTiff

def segmentationcohortnnunet(args=None) :
    SegmentationCohortNNUNet.runfromargumentparser(args)

def segmentationcohortdeepcell(args=None) :
    SegmentationCohortDeepCell.runfromargumentparser(args)

def segmentationcohortmesmerwithihc(args=None) :
    SegmentationCohortMesmerWithIHC.runfromargumentparser(args)

def segmentationcohortmesmercomponenttiff(args=None) :
    SegmentationCohortMesmerComponentTiff.runfromargumentparser(args)
