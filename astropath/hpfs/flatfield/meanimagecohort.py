#imports
from .meanimagesample import MeanImageSample
from .config import CONST
from ...shared.cohort import Im3Cohort, WorkflowCohort
from ...utilities.config import CONST as UNIV_CONST
import pathlib

class MeanImageCohort(Im3Cohort,WorkflowCohort) :
    sampleclass = MeanImageSample
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,filetype='raw',workingdir=pathlib.Path(UNIV_CONST.MEANIMAGE_DIRNAME),et_offset_file=None,skip_masking=False,n_threads=CONST.DEFAULT_N_THREADS,**kwargs) :
        super().__init__(*args,**kwargs)
        self.filetype = filetype
        self.workingdir = workingdir
        self.et_offset_file = et_offset_file
        self.skip_masking = skip_masking
        self.n_threads = n_threads

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        cls.sampleclass.addargumentstoparser(p)
        return p

    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        return {**super().initkwargsfromargumentparser(parsed_args_dict),
                'filetype': parsed_args_dict.pop('filetype'), 
                'workingdir': parsed_args_dict.pop('workingdir'),
                'et_offset_file': None if parsed_args_dict.pop('skip_exposure_time_correction') else parsed_args_dict.pop('exposure_time_offset_file'),
                'skip_masking': parsed_args_dict.pop('skip_masking'),
                'n_threads': parsed_args_dict.pop('n_threads'),
               }

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
                'filetype':self.filetype,
                'workingdir':self.workingdir,
                'et_offset_file':self.et_offset_file,
                'skip_masking':self.skip_masking,
                'n_threads':self.n_threads
               }

    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':self.skip_masking}

def main(args=None):
    MeanImageCohort.runfromargumentparser(args)

if __name__ == "__main__":
    main()
