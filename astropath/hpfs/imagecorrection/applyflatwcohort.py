#imports 
from ...shared.argumentparser import WorkingDirArgumentParser
from ...shared.cohort import CorrectedImageCohort, SelectRectanglesCohort, ParallelCohort
from ...shared.cohort import WorkflowCohort, SelectLayersCohort
from .config import IMAGECORRECTION_CONST
from .applyflatwsample import ApplyFlatWSample

class ApplyFlatWCohort(CorrectedImageCohort, SelectRectanglesCohort, ParallelCohort, WorkflowCohort, 
                            SelectLayersCohort, WorkingDirArgumentParser) :
    sampleclass = ApplyFlatWSample
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,workingdir=None,**kwargs) :
        super().__init__(*args,**kwargs)
        self.workingdir = workingdir
        if self.im3filetype != 'raw':
            raise ValueError('only ever run image correction on raw files')

    @property
    def initiatesamplekwargs(self) :
        to_return = super().initiatesamplekwargs
        to_return['workingdir']=self.workingdir
        to_return['layers']=self.layers
        to_return['skip_et_corrections']=True # Never apply corrections for exposure time
        # Give the correction model file by default
        if to_return['correction_model_file'] is None :
            to_return['correction_model_file']=IMAGECORRECTION_CONST.DEFAULT_CORRECTION_MODEL_FILEPATH
        return to_return

    @property
    def workflowkwargs(self) :
        return {
            **super().workflowkwargs,
            'layers':self.layers,
            'workingdir':self.workingdir,
        }

    @classmethod
    def defaultim3filetype(cls) :
        return 'raw'

def main(args=None) :
    ApplyFlatWCohort.runfromargumentparser(args)

if __name__ == '__main__' :
    main()
