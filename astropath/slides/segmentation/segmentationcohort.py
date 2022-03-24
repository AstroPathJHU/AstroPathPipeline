#imports
from ...shared.argumentparser import SegmentationFolderArgumentParser
from ...shared.cohort import ParallelCohort, SegmentationFolderCohort, orkflowCohort
from .segmentationsample import SegmentationSampleBase, SegmentationSampleNNUNet
from .segmentationsample import SegmentationSampleDeepCell, SegmentationSampleMesmer

class SegmentationCohortBase(ParallelCohort,WorkflowCohort,SegmentationFolderCohort) :
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
