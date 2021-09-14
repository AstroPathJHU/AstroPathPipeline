#imports 
from .imagecorrectionsample import ImageCorrectionSample
from ...shared.argumentparser import WorkingDirArgumentParser
from ...shared.cohort import CorrectedImageCohort, SelectRectanglesCohort, ParallelCohort
from ...shared.cohort import WorkflowCohort, SelectLayersCohort

class ImageCorrectionCohort(CorrectedImageCohort, SelectRectanglesCohort, ParallelCohort, WorkflowCohort, 
                            SelectLayersCohort, WorkingDirArgumentParser) :
    sampleclass = ImageCorrectionSample
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,workingdir=None,**kwargs) :
        super().__init__(*args,**kwargs)
        self.workingdir = workingdir

    @property
    def initiatesamplekwargs(self) :
        to_return = super().initiatesamplekwargs
        to_return['workingdir']=self.workingdir
        to_return['layers']=self.layers
        to_return['filetype']='raw' # only ever run image correction on raw files
        to_return['skip_et_corrections']=True # Never apply corrections for exposure time
        return to_return

    @property
    def workflowkwargs(self) :
        return {
            **super().workflowkwargs,
            'layers':self.layers,
            'workingdir':self.workingdir,
        }

def main(args=None) :
    ImageCorrectionCohort.runfromargumentparser(args)

if __name__ == '__main__' :
    main()