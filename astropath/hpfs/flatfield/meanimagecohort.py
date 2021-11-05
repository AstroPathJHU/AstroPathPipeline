#imports
from .meanimagesample import MeanImageSample
from ...shared.argumentparser import FileTypeArgumentParser, WorkingDirArgumentParser
from ...shared.cohort import CorrectedImageCohort, SelectRectanglesCohort, MaskCohort, ParallelCohort, WorkflowCohort

class MeanImageCohort(CorrectedImageCohort, SelectRectanglesCohort, MaskCohort, ParallelCohort, 
                      WorkflowCohort, WorkingDirArgumentParser, FileTypeArgumentParser) :
    sampleclass = MeanImageSample
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,filetype='raw',workingdir=None,skip_masking=False,**kwargs) :
        super().__init__(*args,**kwargs)
        self.filetype = filetype
        self.workingdir=workingdir
        self.skip_masking = skip_masking

    @classmethod
    def makeargumentparser(cls) :
        p = super().makeargumentparser()
        p.add_argument('--skip-masking', action='store_true',
                       help='''Add this flag to entirely skip masking out the background regions of the images 
                               as they get added [use this argument to completely skip the background thresholding 
                               and masking]''')
        return p

    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict) :
        return {**super().initkwargsfromargumentparser(parsed_args_dict),
                'skip_masking': parsed_args_dict.pop('skip_masking'),
               }

    @property
    def initiatesamplekwargs(self) :
        rd = {**super().initiatesamplekwargs,
              'filetype':self.filetype,
              'skip_masking':self.skip_masking,
            }
        if self.workingdir is not None :
            rd['workingdir'] = self.workingdir
        return rd

    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':self.skip_masking}

def main(args=None) :
    MeanImageCohort.runfromargumentparser(args)

if __name__ == "__main__":
    main()
