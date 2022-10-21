#imports
import pathlib
from ...utilities.miscfileio import cd
from ...utilities.tableio import writetable
from ...shared.argumentparser import FileTypeArgumentParser
from ...shared.cohort import CorrectedImageCohort, WorkflowCohort
from .config import CONST
from .flatfield import FlatfieldComponentTiff, FlatfieldIm3
from .correctedmeanimage import CorrectedMeanImageComponentTiff, CorrectedMeanImageIm3
from .appliedflatfieldsample import AppliedFlatfieldSampleBase, AppliedFlatfieldSampleComponentTiff, AppliedFlatfieldSampleIm3

class AppliedFlatfieldCohortBase(WorkflowCohort) :
    """
    Class to use in investigating the effects of applying flatfield corrections within a cohort
    Each sample in the cohort will have its tissue bulk rectangles randomly split in two. 
    One half is used to calculate flatfield corrections, and the corrections are applied to a meanimage 
    calculated using the other half. The main output is a document describing the effects of applying the corrections. 
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,workingdir,skip_masking=False,image_set_split='random',**kwargs) :
        """
        workingdir = Path to a directory that will hold the results
        """
        super().__init__(*args,**kwargs)
        self.__workingdir = workingdir
        self.__skip_masking = skip_masking
        self.__image_set_split = image_set_split
        self.__metadata_summaries_ff = []
        self.__metadata_summaries_cmi = []
        self.__field_logs_ff = []
        self.__field_logs_cmi = []
        #the next two variables need to be set in subclasses
        self._flatfield = None
        self._corrected_meanimage=None

    def run(self,**kwargs) :
        #run all of the samples individually first like any other cohort (just checks that files exist)
        super().run(**kwargs)
        #after all of the samples have run individually
        with self.globallogger() as logger :
            self._flatfield.logger = logger
            self._corrected_meanimage.logger = logger
            #derive the corrections
            logger.info('Creating flatfield model....')
            self._flatfield.create_flatfield_model()
            #create the mean image that will be corrected
            logger.info('Creating mean image to correct....')
            self._corrected_meanimage.make_mean_image()
            #correct the meanimage with the flatfield model
            logger.info('Applying flatfield corrections to meanimage....')
            self._corrected_meanimage.apply_flatfield_corrections(self._flatfield)
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
                    writetable(f"{CONST.FIELDS_USED_CSV_FILENAME.rstrip('.csv')}_corrected_mean_image.csv",
                                self.__field_logs_cmi)
            #write the output from the corrected mean image
            logger.info('Creating plots and writing output for corrected mean image....')
            samp = None
            for sample in self.samples() :
                if samp is None :
                    samp = sample
                    break
            self._corrected_meanimage.write_output(samp,self.__workingdir)

    def runsample(self,sample,**kwargs) :
        """
        Add the sample's meanimage and mask stack to the batch flatfield meanimage and collect its metadata
        """
        self._flatfield.logger = sample.logger
        self._corrected_meanimage.logger = sample.logger
        #make sure the sample has enough rectangles in the bulk of the tissue to be used
        if len(sample.flatfield_rectangles)<1 or len(sample.meanimage_rectangles)<1 :
            msg = f'{sample.SlideID} only has {len(sample.tissue_bulk_rects)} images in the bulk of the tissue '
            msg+= 'and so it will be ignored in the AppliedFlatfieldCohort.'
            sample.logger.info(msg)
            return
        #run the sample to find or create its masking files if necessary
        super().runsample(sample,**kwargs)
        #add half the rectangles to the flatfield model
        msg = f'{sample.SlideID} will add {len(sample.flatfield_rectangles)} images to the flatfield model'
        sample.logger.info(msg)
        new_field_logs = self._flatfield.stack_rectangle_images(sample,sample.flatfield_rectangles,sample.med_ets,
                                                                 sample.image_masking_dirpath)
        self.__field_logs_ff+=new_field_logs
        self.__metadata_summaries_ff.append(sample.flatfield_metadata_summary)
        #add the other half of the rectangles to the corrected mean image
        msg = f'{sample.SlideID} will add {len(sample.meanimage_rectangles)} images to the corrected mean image'
        sample.logger.info(msg)
        new_field_logs = self._corrected_meanimage.stack_rectangle_images(sample,sample.meanimage_rectangles,
                                                                           sample.med_ets,
                                                                           sample.image_masking_dirpath)
        self.__field_logs_cmi+=new_field_logs
        self.__metadata_summaries_cmi.append(sample.corrected_meanimage_metadata_summary)

    #################### CLASS VARIABLES + PROPERTIES ####################

    sampleclass = AppliedFlatfieldSampleBase

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
                'workingdir':self.__workingdir,
                'skip_masking':self.__skip_masking,
                'image_set_split':self.__image_set_split,
               }

    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':self.__skip_masking}

    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls, **kwargs):
        p = super().makeargumentparser(**kwargs)
        p.add_argument('workingdir', type=pathlib.Path, help='Path to the directory that should hold the results')
        p.add_argument('--skip-masking', action='store_true',
                       help='''Add this flag to entirely skip masking out the background regions 
                                of the images as they get added''')
        p.add_argument('--image-set-split',choices=['random','sequential'],default='random',
                       help='''Whether to split the set of all images into subgroups randomly or sequentially 
                               (default is random)''')
        return p

    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        parsed_args_dict['skip_finished']=False #always rerun the samples, they don't produce any output
        return {**super().initkwargsfromargumentparser(parsed_args_dict),
                'workingdir': parsed_args_dict.pop('workingdir'),
                'skip_masking': parsed_args_dict.pop('skip_masking'),
                'image_set_split': parsed_args_dict.pop('image_set_split'),
               }

class AppliedFlatfieldCohortComponentTiff(AppliedFlatfieldCohortBase) :
    """
    Class for testing the impact of flatfielding on component tiff files
    """

    sampleclass = AppliedFlatfieldSampleComponentTiff

    def __init__(self,*args,**kwargs) :
        """
        workingdir = Path to a directory that will hold the results
        """
        super().__init__(*args,**kwargs)
        #figure out the image dimensions to give to the flatfield and corrected mean image
        image_dimensions = None
        for sample in self.samples() :
            if len(sample.rectangles)>0 :
                with sample.rectangles[0].using_component_tiff() as im :
                    image_dimensions = (im.shape[0],im.shape[1],im.shape[2])
                break
        if image_dimensions is None :
            errmsg = f"ERROR: could not set image dimensions from any sample's rectangles in {self.__class__.__name__}"
            raise RuntimeError(errmsg)
        self._flatfield = FlatfieldComponentTiff(image_dimensions,self.logger)
        self._corrected_meanimage = CorrectedMeanImageComponentTiff(image_dimensions,self.logger)

class AppliedFlatfieldCohortIm3(AppliedFlatfieldCohortBase,CorrectedImageCohort,FileTypeArgumentParser) :
    """
    Class for testing the impact of flatfielding on im3 files
    """

    sampleclass = AppliedFlatfieldSampleIm3

    def __init__(self,*args,filetype='raw',**kwargs) :
        """
        workingdir = Path to a directory that will hold the results
        """
        super().__init__(*args,**kwargs)
        self.__filetype = filetype
        #figure out the image dimensions to give to the flatfield and corrected mean image
        for sample in self.samples() :
            if len(sample.rectangles)>0 :
                image_dimensions = sample.rectangles[0].im3shape
                break
        self._flatfield = FlatfieldIm3(image_dimensions,self.logger)
        self._corrected_meanimage = CorrectedMeanImageIm3(image_dimensions,self.logger)

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
                'filetype':self.__filetype,
               }

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    AppliedFlatfieldCohortIm3.runfromargumentparser(args)

def appliedflatfieldcohortcomponenttiff(args=None) :
    AppliedFlatfieldCohortComponentTiff.runfromargumentparser(args)

if __name__=='__main__' :
    main()
