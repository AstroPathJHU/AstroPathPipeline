#imports
from ...shared.argumentparser import WorkingDirArgumentParser
from ...shared.cohort import ParallelCohort, WorkflowCohort
from .segmentationsample import SegmentationSampleBase, SegmentationSampleNNUNet, SegmentationSampleDeepCell

class SegmentationCohortBase(ParallelCohort,WorkflowCohort,WorkingDirArgumentParser) :
    sampleclass = SegmentationSampleBase
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,workingdir=None,algorithm='nnunet',**kwargs) :
        super().__init__(*args,**kwargs)
        self.workingdir = workingdir
        self.algorithm = algorithm

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
            'workingdir':self.workingdir,
            }

    @property
    def workflowkwargs(self) :
        return {**super().workflowkwargs,
            'workingdir':self.workingdir,
            'algorithm':self.algorithm,
        }

class SegmentationCohortNNUNet(SegmentationCohortBase) :
    sampleclass = SegmentationSampleNNUNet
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,algorithm='nnunet',**kwargs)

class SegmentationCohortDeepCell(SegmentationCohortBase) :
    sampleclass = SegmentationSampleDeepCell
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,algorithm='deepcell',**kwargs)

def segmentationcohortnnunet(args=None) :
    SegmentationCohortNNUNet.runfromargumentparser(args)

def segmentationcohortdeepcell(args=None) :
    SegmentationCohortDeepCell.runfromargumentparser(args)
