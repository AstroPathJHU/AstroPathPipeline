#imports
from ...shared.argumentparser import WorkingDirArgumentParser
from ...shared.cohort import ParallelCohort, WorkflowCohort
from .segmentationsample import SegmentationSampleBase, SegmentationSampleNNUNet, SegmentationSampleDeepCell

class SegmentationCohortBase(ParallelCohort,WorkflowCohort,WorkingDirArgumentParser) :
    sampleclass = SegmentationSampleBase
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,workingdir=None,**kwargs) :
        super().__init__(*args,**kwargs)
        self.workingdir = workingdir

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
            'workingdir':self.workingdir,
            }

    @property
    def workflowkwargs(self) :
        return {**super().workflowkwargs,
            'workingdir':self.workingdir,
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
