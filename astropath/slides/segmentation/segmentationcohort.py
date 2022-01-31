#imports
from ...shared.argumentparser import SegmentationAlgorithmArgumentParser,WorkingDirArgumentParser
from ...shared.cohort import ParallelCohort, WorkflowCohort
from .segmentationsample import SegmentationSample

class SegmentationCohort(ParallelCohort,WorkflowCohort,SegmentationAlgorithmArgumentParser,WorkingDirArgumentParser) :
    sampleclass = SegmentationSample
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,workingdir=None,algorithm='nnunet',**kwargs) :
        super().__init__(*args,**kwargs)
        self.workingdir = workingdir
        self.algorithm = algorithm

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
            'workingdir':self.workingdir,
            'algorithm':self.algorithm,
            }

    @property
    def workflowkwargs(self) :
        return {**super().workflowkwargs,
            'workingdir':self.workingdir,
            'algorithm':self.algorithm,
        }

def main(args=None) :
    SegmentationCohort.runfromargumentparser(args)

if __name__ == "__main__":
    main()