#imports
from .meanimagesample import MeanImageSampleBase
from .meanimage import CorrectedMeanImage
from .flatfield import Flatfield
from ...shared.cohort import Im3Cohort, WorkflowCohort

class AppliedFlatfieldSample(MeanImageSampleBase) :
    """
    Class to use in running most of the MeanImageSample functions but handling the output differently
    """

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.imagefile for r in self.rectangles),
               ]

    def run(self) :
        """
        Main "run" function to be looped when entire cohorts are run
        """
        if not self.skip_masking :
            self.create_or_find_image_masks()
        #make the mean image from all of the tissue bulk rectangles
        new_field_logs = self.__meanimage.stack_images(self.tissue_bulk_rects,self.med_ets,self.image_masking_dirpath)
        for fl in new_field_logs :
            fl.slide = self.SlideID
            self.field_logs.append(fl)
        bulk_rect_ts = [r.t for r in self.tissue_bulk_rects]
        with cd(self.workingdirpath) :
            writetable(self.workingdirpath / f'{self.SlideID}-{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}',
                      [MetadataSummary(self.SlideID,self.Project,self.Cohort,self.microscopename,str(min(bulk_rect_ts)),str(max(bulk_rect_ts)))])
        #create and write out the final mask stack, mean image, and std. error of the mean image
        self.__meanimage.write_output(self.SlideID,self.workingdirpath)
        #write out the field log
        with cd(self.workingdirpath) :
            writetable(CONST.FIELDS_USED_CSV_FILENAME,self.field_logs)

    @classmethod
    def getoutputfiles(cls,**kwargs) :
        return [*super().getoutputfiles(**kwargs)]
    @classmethod
    def logmodule(cls) : 
        return "appliedflatfield"
    @classmethod
    def workflowdependencyclasses(cls):
        return super().workflowdependencyclasses()

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
        self.__filetype = filetype
        self.__et_offset_file = et_offset_file
        self.__skip_masking = skip_masking
        self.__flatfield = Flatfield(self.logger)
        self.__corrected_meanimage = CorrectedMeanImage(self.logger)

    def run(self,**kwargs) :
        #run all of the samples individually first like any other cohort (just checks that files exist)
        super().run(**kwargs)
        #after all of the samples have run individually
        with self.globallogger() as logger :
            #derive the corrections
            logger.info('Creating flatfield model....')
            self.__flatfield.create_flatfield_model()
            #correct the meanimage with the flatfield model
            logger.info('Applying flatfield corrections to meanimage....')
            self.__corrected_meanimage.apply_flatfield_corrections(self.__flatfield)
            #write the output from the corrected mean image
            logger.info('Creating plots and writing output....')
            self.__corrected_meanimage.write_output(self.__workingdir)

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
                'workingdir':self.__workingdir,
                'filetype':self.__filetype,
                'et_offset_file':self.__et_offset_file,
                'skip_masking':self.__skip_masking,
               }

    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':self.skip_masking}

    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        p.add_argument('workingdir', type=pathlib.Path, help='path to the working directory where results will be saved')
        cls.sampleclass.addargumentstoparser(p)
        return p

    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        parsed_args_dict['skip_finished']=False
        return {**super().initkwargsfromargumentparser(parsed_args_dict),
                'filetype': parsed_args_dict.pop('filetype'), 
                'et_offset_file': None if parsed_args_dict.pop('skip_exposure_time_correction') else parsed_args_dict.pop('exposure_time_offset_file'),
                'skip_masking': parsed_args_dict.pop('skip_masking'),
               }