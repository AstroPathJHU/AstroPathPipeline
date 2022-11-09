#imports
from ...shared.argumentparser import FileTypeArgumentParser, WorkingDirArgumentParser
from ...shared.cohort import CorrectedImageCohort, SelectRectanglesCohort, MaskCohort, ParallelCohort, WorkflowCohort
from .meanimagesample import MeanImageSampleBase, MeanImageSampleComponentTiffTissue, MeanImageSampleIm3Tissue

class MeanImageCohortBase(CorrectedImageCohort, ParallelCohort, MaskCohort, SelectRectanglesCohort, 
                          WorkflowCohort, WorkingDirArgumentParser) :
    sampleclass = MeanImageSampleBase
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,workingdir=None,skip_masking=False,**kwargs) :
        super().__init__(*args,**kwargs)
        self.workingdir=workingdir
        self.skip_masking = skip_masking

    @classmethod
    def makeargumentparser(cls, **kwargs) :
        p = super().makeargumentparser(**kwargs)
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
              'skip_masking':self.skip_masking,
            }
        if self.workingdir is not None :
            rd['workingdir'] = self.workingdir
        return rd

    @property
    def workflowkwargs(self) :
        result = {**super().workflowkwargs,'skip_masking':self.skip_masking}
        if self.workingdir is not None :
            result['workingdir'] = self.workingdir
        return result

class MeanImageCohortComponentTiff(MeanImageCohortBase) :
    sampleclass = MeanImageSampleComponentTiffTissue

class MeanImageCohortIm3(MeanImageCohortBase, FileTypeArgumentParser) :
    sampleclass = MeanImageSampleIm3Tissue

    def __init__(self,*args,filetype='raw',**kwargs) :
        super().__init__(*args,**kwargs)
        self.filetype = filetype

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
                'filetype':self.filetype,
            }

def main(args=None) :
    MeanImageCohortIm3.runfromargumentparser(args)

if __name__ == "__main__":
    main()
