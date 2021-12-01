#imports
from argparse import PARSER
import pathlib, re
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.tableio import readtable
from ...utilities.img_file_io import get_image_hwl_from_xml_file
from ...shared.sample import WorkflowSample
from ...shared.cohort import WorkflowCohort
from ...shared.multicohort import MultiCohortBase
from .config import CONST
from .utilities import ModelTableEntry
from .imagestack import Flatfield
from .meanimagesample import MeanImageSample

class BatchFlatfieldSample(WorkflowSample) :
    """
    Small utility class to hold sample-dependent information for the batch flatfield run
    Just requires as input files the relevant output of the meanimage mode
    """

    multilayer = True

    def __init__(self,*args,meanimage_dirname=UNIV_CONST.MEANIMAGE_DIRNAME,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__meanimage_dirname = meanimage_dirname

    @property
    def meanimagefolder(self) :
        return self.im3folder/self.__meanimage_dirname
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
    def run(self,version,flatfield,samplesprocessed,totalsamples) :
        msg = f'Adding mean image and mask stack from {self.SlideID} to flatfield model version '
        msg+= f'{version} ({len(samplesprocessed)+1} of {totalsamples})....'
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

class BatchFlatfieldCohort(WorkflowCohort) :
    """
    Class to handle combining several samples' meanimages into a single flatfield model for a batch
    (Single-cohort placeholder for BatchFlatfieldMultiCohort)
    """

    sampleclass = BatchFlatfieldSample

    def __init__(self,*args,meanimage_dirname=UNIV_CONST.MEANIMAGE_DIRNAME,**kwargs) :
        super().__init__(*args,**kwargs) 
        self.__meanimage_dirname = meanimage_dirname

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
                'meanimage_dirname':self.__meanimage_dirname,
            }

    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':False}

class BatchFlatfieldMultiCohort(MultiCohortBase):
    """
    Multi-cohort version of batch flatfield code that combines several samples' meanimages 
    into a single flatfield model
    """

    def __init__(self,*args,version,meanimage_dirname=UNIV_CONST.MEANIMAGE_DIRNAME,outdir,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__version = version
        self.__meanimage_dirname = meanimage_dirname
        self.__outdir = outdir

    def run(self, **kwargs):
        totalsamples = 0
        image_dimensions = None
        with self.globallogger() as logger:
            #start up the flatfield after figuring out its dimensions
            for cohort in self.cohorts :
                for sample in cohort.filteredsamples() :
                    dims = get_image_hwl_from_xml_file(sample.root,sample.SlideID)
                    if image_dimensions is None :
                        image_dimensions = dims
                    else :
                        if dims!=image_dimensions :
                            errmsg=f'ERROR: {sample.SlideID} dimensions {dims} mismatched to {image_dimensions}'
                            raise ValueError(errmsg)
                    totalsamples += 1
            if image_dimensions is None :
                raise ValueError("No non-empty samples")
            flatfield = Flatfield(image_dimensions,logger)
            samplesprocessed=[]
            #Run all the samples individually like for a regular MultiCohort
            super().run(flatfield=flatfield, 
                        samplesprocessed=samplesprocessed, 
                        version=self.__version, 
                        totalsamples=totalsamples, **kwargs)
            totalsamples = len(samplesprocessed)
            #actually create the flatfield after all the samples have been added
            logger.info(f'Creating final flatfield model for version {self.__version}....')
            flatfield.create_flatfield_model()
            #write out the flatfield model
            logger.debug(f'Writing out flatfield model, plots, and summary pdf for version {self.__version}....')
            flatfield.write_output(self.__version,self.workingdir)

    #################### CLASS VARIABLES + PROPERTIES ####################

    singlecohortclass = BatchFlatfieldCohort

    @property
    def workingdir(self) :
        return self.__outdir / UNIV_CONST.FLATFIELD_DIRNAME / f'{CONST.FLATFIELD_DIRNAME_STEM}_{self.__version}'

    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        p.add_argument('--version',
                       help="version of the flatfield model that should be created from the given slides' meanimages")
        p.add_argument('--flatfield-model-file',type=pathlib.Path,
                        default=pathlib.Path('//bki04/astropath_processing/AstroPathFlatfieldModels.csv'),
                        help='path to a .csv file defining which slides should be used for the given version')
        p.add_argument('--meanimage_dirname',default=UNIV_CONST.MEANIMAGE_DIRNAME,
                       help='''The name of the meanimage directories to use 
                                (useful if multiple meanimage versions have been created)''')
        p.add_argument('--outdir',type=pathlib.Path,default=pathlib.Path('//bki04/astropath_processing'),
                       help='''directory where the output will be placed (a "flatfield" directory will be created 
                                inside outdir if one does not already exist)''')
        return p
    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        #use the sample regex to choose samples listed in the model file (if no sampleregex was given)
        version = parsed_args_dict.pop('version')
        flatfield_model_file = parsed_args_dict.pop('flatfield_model_file')
        if parsed_args_dict['sampleregex'] is None :
            if not flatfield_model_file.is_file() :
                raise ValueError(f'ERROR: flatfield model file {flatfield_model_file} not found!')
            all_model_table_entries = readtable(flatfield_model_file,ModelTableEntry)
            model_table_entries = [te for te in all_model_table_entries if te.version==version]
            n_entries = len(model_table_entries)
            if n_entries<1 :
                errmsg=f'ERROR: {n_entries} entries found in {flatfield_model_file} for version {version}!'
                raise ValueError(errmsg)
            slide_IDs = [te.SlideID for te in model_table_entries]
            new_regex = '('
            for sid in slide_IDs :
                new_regex+=sid+r'\b|'
            new_regex=new_regex[:-1]+')'
            parsed_args_dict['sampleregex']=re.compile(new_regex)
        #always rerun the samples, they don't produce any output
        parsed_args_dict['skip_finished']=False 
        #return the kwargs dict
        return {
            **super().initkwargsfromargumentparser(parsed_args_dict),
            'version': version,
            'meanimage_dirname': parsed_args_dict.pop('meanimage_dirname'),
            'outdir': parsed_args_dict.pop('outdir'), 
        }

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    BatchFlatfieldMultiCohort.runfromargumentparser(args)

if __name__=='__main__' :
    main()
