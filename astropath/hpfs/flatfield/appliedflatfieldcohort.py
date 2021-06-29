#imports
from .meanimagesample import MeanImageSampleBase
from .meanimage import MeanImage
from .flatfield import Flatfield
from ...shared.cohort import Im3Cohort, WorkflowCohort

class AppliedFlatfieldSample(MeanImageSampleBase) :
    """
    Class to use in running most of the MeanImageSample functions but handling the output differently
    """

    def __init__(self,*args,**kwargs) :
        pass

class AppliedFlatfieldCohort(Im3Cohort,WorkflowCohort) :
    """
    Class to use in investigating the effects of applying flatfield corrections within a cohort
    Each sample in the cohort will have its tissue bulk rectangles randomly split in two. 
    One half is used to calculate flatfield corrections, and the corrections are applied to a meanimage 
    calculated using the other half. The main output is a document describing the effects of applying the corrections. 
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,workingdir,filetype='raw',et_offset_file=None,skip_masking=False,**kwargs) :
        """
        workingdir = Path to a directory that will hold the results
        """
        super().__init__(*args,**kwargs)
        self.__workingdir = workingdir
        self.__flatfield = Flatfield(self.logger)
        self.__meanimage = MeanImage(self.logger)

    def run(self,**kwargs) :
        #run all of the samples individually first like any other cohort (just checks that files exist)
        super().run(**kwargs)
        #after all of the samples have run individually
        with self.globallogger() as logger :
            #derive and apply the corrections
            logger.info(f'Creating final flatfield model for batch {self.__batchID:02d}....')
            self.__flatfield.create_flatfield_model()
            #write out the flatfield model
            logger.info(f'Writing out flatfield model, plots, and summary pdf for batch {self.__batchID:02d}....')
            self.__flatfield.write_output(self.__batchID,self.workingdir)

    def runsample(self,sample,**kwargs) :
        """
        Add the sample's meanimage and mask stack to the batch flatfield meanimage and collect its metadata
        """
        #running the sample just makes sure that its file exist
        super().runsample(sample,**kwargs)
        #add the sample's information to the flatfield model that's being created
        msg = f'Adding mean image and mask stack from {sample.SlideID} to flatfield model for batch {self.__batchID:02d} '
        msg+= f'({self.__samples_added+1} of {len(list(self.filteredsamples))})....'
        sample.logger.info(msg)
        self.__flatfield.add_sample(sample)
        self.__samples_added+=1

    #################### CLASS VARIABLES + PROPERTIES ####################

    sampleclass = AppliedFlatfieldSample

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

    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        cls.sampleclass.addargumentstoparser(p)
        return p

    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        return {**super().initkwargsfromargumentparser(parsed_args_dict),
                'filetype': parsed_args_dict.pop('filetype'), 
                'et_offset_file': None if parsed_args_dict.pop('skip_exposure_time_correction') else parsed_args_dict.pop('exposure_time_offset_file'),
                'skip_masking': parsed_args_dict.pop('skip_masking'),
               }