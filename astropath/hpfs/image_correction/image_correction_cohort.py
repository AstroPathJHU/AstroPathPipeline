#imports 
from .image_correction_sample import ImageCorrectionSample
from ...shared.argumentparser import WorkingDirArgumentParser
from ...shared.cohort import CorrectedImageCohort, SelectRectanglesCohort, ParallelCohort, WorkflowCohort, SelectLayersCohort

class ImageCorrectionCohort(CorrectedImageCohort, SelectRectanglesCohort, ParallelCohort, WorkflowCohort, SelectLayersCohort, WorkingDirArgumentParser) :
    sampleclass = ImageCorrectionSample
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,workingdir,**kwargs) :
        super().__init__(*args,**kwargs)
        self.workingdir = workingdir

    @property
    def initiatesamplekwargs(self) :
        return {
            **super().initiatesamplekwargs,
            'workingdir':self.workingdir,
            'layers':self.layers,
            }

    @property
    def workflowkwargs(self) :
        return {
            **super().workflowkwargs,
            'layers':self.layers,
        }

def main(args=None) :
    ImageCorrectionCohort.runfromargumentparser(args)

if __name__ == '__main__' :
    main()