#imports
from .meanimagesample import MeanImageSample
from .flatfield import Flatfield
from .config import CONST
from ...shared.sample import WorkflowSample
from ...shared.cohort import WorkflowCohort
from ...utilities.config import CONST as UNIV_CONST

class BatchFlatfieldSample(WorkflowSample) :
    """
    Small utility class to hold sample-dependent information for the batch flatfield run
    Just requires as input files the relevant output of the meanimage mode
    """
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
    @property
    def inputfiles(self,**kwargs) :
        return [self.meanimage,self.sumimagessquared,self.maskstack,self.fieldsused,self.metadatasummary]
    def run(self,**kwargs) :
        pass
    @classmethod
    def getoutputfiles(cls,**kwargs) :
        return []
    @classmethod
    def defaultunits(cls) :
        return MeanImageSample.defaultunits()
    @classmethod
    def logmodule(cls) : 
        return "batchflatfield"
    @classmethod
    def workflowdependencyclasses(cls):
        return super().workflowdependencyclasses()

class BatchFlatfieldCohort(WorkflowCohort) :
    """
    Class to handle combining several samples' meanimages into a single flatfield model for a batch
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,batchID=-1,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__batchID = batchID
        #start up the flatfield
        self.__flatfield = Flatfield(self.logger)

    def run(self,**kwargs) :
        #run all of the samples individually first like any other cohort
        super().run(**kwargs)
        #actually create the flatfield after all the samples have been added
        self.__flatfield.create_flatfield_model()
        #write out the flatfield model
        self.__flatfield.write_output(self.__batchID,self.workingdir)

    def runsample(self,sample,**kwargs) :
        """
        Add the sample's meanimage and mask stack to the batch flatfield meanimage and collect its metadata
        """
        self.__flatfield.add_sample(sample)

    #################### CLASS VARIABLES + PROPERTIES ####################

    sampleclass = BatchFlatfieldSample

    @property
    def workingdir(self) :
        return self.root / CONST.TOP_FLATFIELD_DIRNAME / f'{CONST.FLATFIELD_DIRNAME_STEM}{self.__batchID:02d}'

    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        p.add_argument('--batchID',type=int,default=-1,help='BatchID for the flatfield model created from the given list of slideIDs')
        return p
    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        return {
            **super().initkwargsfromargumentparser(parsed_args_dict),
            'batchID': parsed_args_dict.pop('batchID'), 
        }

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    BatchFlatfieldCohort.runfromargumentparser(args)

if __name__=='__main__' :
    main()
