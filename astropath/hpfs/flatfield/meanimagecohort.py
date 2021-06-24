#imports
from .meanimagesample import MeanImageSample
from .config import CONST
from ...shared.cohort import Im3Cohort, WorkflowCohort

class MeanImageCohort(Im3Cohort,WorkflowCohort) :
    sampleclass = MeanImageSample
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,workingdir=pathlib.Path(UNIV_CONST.MEANIMAGE_DIRNAME),et_offset_file=None,skip_masking=False,n_threads=CONST.DEFAULT_N_THREADS,**kwargs) :
        super().__init__(*args,**kwargs)
        self.workingdir = workingdir
        self.et_offset_file = et_offset_file
        self.skip_masking = skip_masking
        self.n_threads = n_threads

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs(),
                'workingdir':self.workingdir,
                'et_offset_file':self.et_offset_file,
                'skip_masking':self.skip_masking,
                'n_threads':self.n_threads
               }

def main(args=None):
    MeanImageCohort.runfromargumentparser(args)

if __name__ == "__main__":
    main()
