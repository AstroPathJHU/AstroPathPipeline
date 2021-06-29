#imports
from .meanimagesample import MeanImageSampleBase
from .imagestack import CorrectedMeanImage, Flatfield
from .rectangle import RectangleCorrectedIm3MultiLayer
from ...shared.cohort import Im3Cohort, WorkflowCohort
import random

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
        #make sure the sample has enough rectangles in the bulk of the tissue to be used
        if len(sample.tissue_bulk_rects)<2 :
            sample.logger.info(f'{sample.SlideID} only has {len(sample.tissue_bulk_rects)} images in the bulk of the tissue and so it will be ignored in the AppliedFlatfieldCohort.')
            return
        #run the sample to find or create its masking files if necessary
        super().runsample(sample,**kwargs)
        #split the set of sample rectangles in the bulk of the tissue in half randomly
        flatfield_rectangles = random.sample(sample.tissue_bulk_rects,len(sample.tissue_bulk_rects)/2)
        meanimage_rectangles = [r for r in sample.tissue_bulk_rects if r not in flatfield_rectangles]
        #add half the rectangles to the flatfield model
        self.__flatfield.stack_rectangle_images(flatfield_rectangles,med_ets,)


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