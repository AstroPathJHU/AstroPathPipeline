#imports
import pathlib
from .meanimagesample import MeanImageSample
from .imagestack import Flatfield
from .config import CONST
from ...shared.sample import ReadRectanglesIm3FromXML, WorkflowSample
from ...shared.cohort import Im3Cohort, WorkflowCohort
from ...shared.multicohort import MultiCohortBase
from ...utilities.config import CONST as UNIV_CONST

class BatchFlatfieldSample(ReadRectanglesIm3FromXML,WorkflowSample) :
    """
    Small utility class to hold sample-dependent information for the batch flatfield run
    Just requires as input files the relevant output of the meanimage mode
    """
    multilayer = True
    @property
    def meanimagefolder(self) :
        return self.im3folder/UNIV_CONST.MEANIMAGE_DIRNAME
    @property
    def meanimage(self) :
        return self.meanimagefolder/f'{self.SlideID}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}'
    @property
    def sumimagessquared(self) :
        return self.meanimagefolder/f'{self.SlideID}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}'
    @property
    def maskstack(self) :
        return self.meanimagefolder/f'{self.SlideID}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}'
    @property
    def fieldsused(self) :
        return self.meanimagefolder/CONST.FIELDS_USED_CSV_FILENAME
    @property
    def metadatasummary(self) :
        return self.meanimagefolder/f'{self.SlideID}-{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}'
    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                self.meanimage,self.sumimagessquared,self.maskstack,self.fieldsused,self.metadatasummary]
    def run(self,batchID,flatfield,samplesprocessed,totalsamples) :
        msg = f'Adding mean image and mask stack from {self.SlideID} to flatfield model for batch '
        msg+= f'{batchID:02d} ({len(samplesprocessed)+1} of {totalsamples})....'
        self.logger.info(msg)
        flatfield.add_batchflatfieldsample(self)
        samplesprocessed.append(self)
    @classmethod
    def getoutputfiles(cls,**kwargs) :
        return [*super().getoutputfiles(**kwargs)]
    @classmethod
    def defaultunits(cls) :
        return MeanImageSample.defaultunits()
    @classmethod
    def logmodule(cls) : 
        return "batchflatfield"
    @classmethod
    def workflowdependencyclasses(cls):
        return [*super().workflowdependencyclasses(),MeanImageSample]

class BatchFlatfieldCohort(Im3Cohort,WorkflowCohort) :
    """
    Class to handle combining several samples' meanimages into a single flatfield model for a batch
    """

    #################### CLASS VARIABLES + PROPERTIES ####################

    sampleclass = BatchFlatfieldSample

    #################### CLASS METHODS ####################

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
                'filetype':'raw',
               }
    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':False}

class BatchFlatfieldMultiCohort(MultiCohortBase):
    def __init__(self,*args,outdir,batchID=-1,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__batchID = batchID
        self.__outdir = outdir

    @property
    def workingdir(self) :
        return self.__outdir / UNIV_CONST.FLATFIELD_DIRNAME / f'{CONST.FLATFIELD_DIRNAME_STEM}{self.__batchID:02d}'

    singlecohortclass = BatchFlatfieldCohort
    def run(self, **kwargs):
        totalsamples = 0
        image_dimensions = None
        with self.globallogger() as logger:
            #start up the flatfield after figuring out its dimensions
            for cohort in self.cohorts :
                for sample in cohort.filteredsamples :
                    if image_dimensions is None and len(sample.rectangles)>0 :
                        image_dimensions = sample.rectangles[0].imageshapeinoutput
                    totalsamples += 1
            if image_dimensions is None :
                raise ValueError("No non-empty samples")

            flatfield = Flatfield(image_dimensions,logger)
            samplesprocessed=[]

            super().run(flatfield=flatfield, samplesprocessed=samplesprocessed, batchID=self.__batchID, totalsamples=totalsamples, **kwargs)

            totalsamples = len(samplesprocessed)

            #actually create the flatfield after all the samples have been added
            logger.info(f'Creating final flatfield model for batch {self.__batchID:02d}....')
            flatfield.create_flatfield_model()
            #write out the flatfield model
            logger.info(f'Writing out flatfield model, plots, and summary pdf for batch {self.__batchID:02d}....')
            flatfield.write_output(self.__batchID,self.workingdir)

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        p.add_argument('--batchID',type=int,default=-1,
                       help='BatchID for the flatfield model created from the given list of slideIDs')
        p.add_argument('--outdir',type=pathlib.Path,required=True,
                       help='directory where the output will be placed')
        return p
    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        parsed_args_dict['skip_finished']=False #always rerun the samples, they don't produce any output
        return {
            **super().initkwargsfromargumentparser(parsed_args_dict),
            'batchID': parsed_args_dict.pop('batchID'), 
            'outdir': parsed_args_dict.pop('outdir'), 
        }

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    BatchFlatfieldMultiCohort.runfromargumentparser(args)

if __name__=='__main__' :
    main()
