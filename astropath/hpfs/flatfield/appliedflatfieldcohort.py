#imports
from .meanimagesample import MeanImageSampleBase
from .imagestack import CorrectedMeanImage, Flatfield
from .config import CONST
from ...shared.argumentparser import FileTypeArgumentParser, ImageCorrectionArgumentParser
from ...shared.cohort import Im3Cohort, WorkflowCohort
from ...shared.sample import WorkflowSample
from ...shared.rectangle import RectangleCorrectedIm3MultiLayer
from ...shared.overlap import Overlap
from ...utilities.tableio import writetable
from ...utilities.misc import cd, MetadataSummary
from ...utilities.config import CONST as UNIV_CONST
import random, pathlib

class AppliedFlatfieldSample(MeanImageSampleBase,WorkflowSample) :
    """
    Class to use in running most of the MeanImageSample functions but handling the output differently
    """

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__metadata_summary_ff = None
        self.__metadata_summary_cmi = None
        if len(self.tissue_bulk_rects)>1 :
            self.__flatfield_rectangles = random.sample(self.tissue_bulk_rects,round(len(self.tissue_bulk_rects)/2))
            self.__meanimage_rectangles = [r for r in self.tissue_bulk_rects if r not in self.__flatfield_rectangles]
        else :
            self.__flatfield_rectangles = []
            self.__meanimage_rectangles = []

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.imagefile for r in self.rectangles),
               ]

    def run(self) :
        #find or create the masks for the sample if they're needed
        if not self.skip_masking :
            self.create_or_find_image_masks()
        #set the metadata summaries for the sample
        ff_rect_ts = [r.t for r in self.__flatfield_rectangles]
        cmi_rect_ts = [r.t for r in self.__meanimage_rectangles]
        self.__metadata_summary_ff = MetadataSummary(self.SlideID,self.Project,self.Cohort,self.microscopename,str(min(ff_rect_ts)),str(max(ff_rect_ts)))
        self.__metadata_summary_cmi = MetadataSummary(self.SlideID,self.Project,self.Cohort,self.microscopename,str(min(cmi_rect_ts)),str(max(cmi_rect_ts)))

    @property
    def flatfield_rectangles(self) :
        return self.__flatfield_rectangles
    @property
    def meanimage_rectangles(self) :
        return self.__meanimage_rectangles
    @property
    def flatfield_metadata_summary(self) :
        return self.__metadata_summary_ff
    @property
    def corrected_meanimage_metadata_summary(self) :
        return self.__metadata_summary_cmi

    multilayer = True
    rectangletype = RectangleCorrectedIm3MultiLayer
    overlaptype = Overlap
    nclip = UNIV_CONST.N_CLIP

    @classmethod
    def getoutputfiles(cls,**kwargs) :
        return [*super().getoutputfiles(**kwargs)]
    @classmethod
    def logmodule(cls) : 
        return "appliedflatfield"
    @classmethod
    def workflowdependencyclasses(cls):
        return super().workflowdependencyclasses()

class AppliedFlatfieldCohort(Im3Cohort, WorkflowCohort, FileTypeArgumentParser, ImageCorrectionArgumentParser) :
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
        self.__metadata_summaries_ff = []
        self.__metadata_summaries_cmi = []
        self.__field_logs_ff = []
        self.__field_logs_cmi = []

    def run(self,**kwargs) :
        #run all of the samples individually first like any other cohort (just checks that files exist)
        super().run(**kwargs)
        #after all of the samples have run individually
        with self.globallogger() as logger :
            self.__flatfield.logger = logger
            self.__corrected_meanimage.logger = logger
            #derive the corrections
            logger.info('Creating flatfield model....')
            self.__flatfield.create_flatfield_model()
            #create the mean image that will be corrected
            logger.info('Creating mean image to correct....')
            self.__corrected_meanimage.make_mean_image()
            #correct the meanimage with the flatfield model
            logger.info('Applying flatfield corrections to meanimage....')
            self.__corrected_meanimage.apply_flatfield_corrections(self.__flatfield)
            #save the metadata summaries and the field logs
            logger.info('Writing out metadata summaries and field logs....')
            if not self.__workingdir.is_dir() :
                self.__workingdir.mkdir(parents=True)
            if len(self.__metadata_summaries_ff)>0 :
                with cd(self.__workingdir) :
                    writetable('metadata_summary_flatfield_stacked_images.csv',self.__metadata_summaries_ff)
            if len(self.__metadata_summaries_cmi)>0 :
                with cd(self.__workingdir) :
                    writetable('metadata_summary_corrected_mean_image_stacked_images.csv',self.__metadata_summaries_cmi)
            if len(self.__field_logs_ff)>0 :
                with cd(self.__workingdir) :
                    writetable(f"{CONST.FIELDS_USED_CSV_FILENAME.rstrip('.csv')}_flatfield.csv",self.__field_logs_ff)
            if len(self.__field_logs_cmi)>0 :
                with cd(self.__workingdir) :
                    writetable(f"{CONST.FIELDS_USED_CSV_FILENAME.rstrip('.csv')}_corrected_mean_image.csv",self.__field_logs_cmi)
            #write the output from the corrected mean image
            logger.info('Creating plots and writing output for corrected mean image....')
            self.__corrected_meanimage.write_output(self.__workingdir)

    def runsample(self,sample,**kwargs) :
        """
        Add the sample's meanimage and mask stack to the batch flatfield meanimage and collect its metadata
        """
        self.__flatfield.logger = sample.logger
        self.__corrected_meanimage.logger = sample.logger
        #make sure the sample has enough rectangles in the bulk of the tissue to be used
        if len(sample.flatfield_rectangles)<1 or len(sample.meanimage_rectangles)<1 :
            sample.logger.info(f'{sample.SlideID} only has {len(sample.tissue_bulk_rects)} images in the bulk of the tissue and so it will be ignored in the AppliedFlatfieldCohort.')
            return
        #run the sample to find or create its masking files if necessary
        super().runsample(sample,**kwargs)
        #add half the rectangles to the flatfield model
        sample.logger.info(f'{sample.SlideID} will add {len(sample.flatfield_rectangles)} images to the flatfield model')
        new_field_logs = self.__flatfield.stack_rectangle_images(sample.flatfield_rectangles,sample.med_ets,sample.image_masking_dirpath)
        self.__field_logs_ff+=new_field_logs
        self.__metadata_summaries_ff.append(sample.flatfield_metadata_summary)
        #add the other half of the rectangles to the corrected mean image
        sample.logger.info(f'{sample.SlideID} will add {len(sample.meanimage_rectangles)} images to the corrected mean image')
        new_field_logs = self.__corrected_meanimage.stack_rectangle_images(sample.meanimage_rectangles,sample.med_ets,sample.image_masking_dirpath)
        self.__field_logs_cmi+=new_field_logs
        self.__metadata_summaries_cmi.append(sample.corrected_meanimage_metadata_summary)

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
        return{**super().workflowkwargs,'skip_masking':self.__skip_masking}

    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        p.add_argument('workingdir', type=pathlib.Path, help='Path to the directory that should hold the results')
        p.add_argument('--skip_masking', action='store_true',
                       help='Add this flag to entirely skip masking out the background regions of the images as they get added')
        return p

    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        parsed_args_dict['skip_finished']=False #always rerun the samples, they don't produce any output
        return {**super().initkwargsfromargumentparser(parsed_args_dict),
                'workingdir': parsed_args_dict.pop('workingdir'),
                'skip_masking': parsed_args_dict.pop('skip_masking'),
               }

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    AppliedFlatfieldCohort.runfromargumentparser(args)

if __name__=='__main__' :
    main()
