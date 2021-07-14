#imports
from .meanimagesample import MeanImageSample
from ...shared.argumentparser import FileTypeArgumentParser, ImageCorrectionArgumentParser
from ...shared.cohort import Im3Cohort, ParallelCohort, WorkflowCohort

class MeanImageCohort(Im3Cohort, ParallelCohort, WorkflowCohort, FileTypeArgumentParser, ImageCorrectionArgumentParser) :
    sampleclass = MeanImageSample
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,filetype='raw',et_offset_file=None,skip_masking=False,**kwargs) :
        super().__init__(*args,**kwargs)
        self.filetype = filetype
        self.et_offset_file = et_offset_file
        self.skip_masking = skip_masking

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        p.add_argument('--skip_masking', action='store_true',
                       help='''Add this flag to entirely skip masking out the background regions of the images as they get added
                       [use this argument to completely skip the background thresholding and masking]''')
        return p

    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        return {**super().initkwargsfromargumentparser(parsed_args_dict),
                'skip_masking': parsed_args_dict.pop('skip_masking'),
               }

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
                'filetype':self.filetype,
                'et_offset_file':self.et_offset_file,
                'skip_masking':self.skip_masking,
               }

    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':self.skip_masking}

def main(args=None):
    MeanImageCohort.runfromargumentparser(args)

if __name__ == "__main__":
    main()
