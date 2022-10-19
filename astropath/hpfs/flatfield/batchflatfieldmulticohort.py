#imports
import pathlib, re
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.tableio import readtable
from ...utilities.img_file_io import get_image_hwl_from_xml_file
from ...shared.cohort import WorkflowCohort
from ...shared.multicohort import MultiCohortBase
from .config import CONST
from .utilities import FlatfieldModelTableEntry
from .flatfield import FlatfieldComponentTiff, FlatfieldIm3
from .batchflatfieldsample import BatchFlatfieldSampleBase, BatchFlatfieldSampleComponentTiff, BatchFlatfieldSampleIm3

class BatchFlatfieldCohortBase(WorkflowCohort) :
    """
    Base class for combining several samples' mean images into a flatfield model
    """

    sampleclass = BatchFlatfieldSampleBase

    def __init__(self,*args,meanimage_dirname,version,outdir,**kwargs) :
        super().__init__(*args,**kwargs) 
        self.__meanimage_dirname = meanimage_dirname
        self.__version = version
        self.__outdir = outdir

    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
                'meanimage_dirname':self.__meanimage_dirname,
                'version':self.__version,
            }

    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':False}

class BatchFlatfieldCohortComponentTiff(BatchFlatfieldCohortBase) :
    """
    Class to handle combining several samples' component tiff meanimages into a single flatfield model for a batch
    """

    sampleclass = BatchFlatfieldSampleComponentTiff

    def __init__(self,*args,meanimage_dirname=f'{UNIV_CONST.MEANIMAGE_DIRNAME}_comp_tiff',**kwargs) :
        super().__init__(*args,meanimage_dirname=meanimage_dirname,**kwargs) 

class BatchFlatfieldCohortIm3(BatchFlatfieldCohortBase) :
    """
    Class to handle combining several samples' im3 meanimages into a single flatfield model for a batch
    """

    sampleclass = BatchFlatfieldSampleIm3

    def __init__(self,*args,meanimage_dirname=UNIV_CONST.MEANIMAGE_DIRNAME,**kwargs) :
        super().__init__(*args,meanimage_dirname=meanimage_dirname,**kwargs) 

class BatchFlatfieldMultiCohortBase(MultiCohortBase):
    """
    Multi-cohort version of batch flatfield code that combines several samples' meanimages 
    into a single flatfield model
    """

    def __init__(self,*args,version,outdir,**kwargs) :
        super().__init__(*args,version=version,outdir=outdir,**kwargs)
        self.__outdir = outdir
        self.__version = version

    def run(self, **kwargs):
        totalsamples = 0
        image_dimensions = None
        with self.globallogger(SlideID=self.__version) as logger:
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
            flatfield = self.flatfield_type(image_dimensions,logger)
            samplesprocessed=[]
            #Run all the samples individually like for a regular MultiCohort
            super().run(flatfield=flatfield, 
                        samplesprocessed=samplesprocessed, 
                        totalsamples=totalsamples, **kwargs)
            totalsamples = len(samplesprocessed)
            #actually create the flatfield after all the samples have been added
            logger.info(f'Creating final flatfield model for version {self.__version}....')
            flatfield.create_flatfield_model()
            #write out the flatfield model
            logger.debug(f'Writing out flatfield model, plots, and summary pdf for version {self.__version}....')
            samp = None
            for cohort in self.cohorts :
                for sample in cohort.samples() :
                    if samp is None :
                        samp = sample
                        break
            flatfield.write_output(samp,self.__version,self.workingdir)

    #################### CLASS VARIABLES + PROPERTIES ####################

    singlecohortclass = None
    flatfield_type = None

    @property
    def workingdir(self) :
        return self.__outdir / UNIV_CONST.FLATFIELD_DIRNAME / f'{UNIV_CONST.FLATFIELD_DIRNAME}_{self.__version}'

    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls, **kwargs):
        p = super().makeargumentparser(**kwargs)
        p.add_argument('--version',
                       help="version of the flatfield model that should be created from the given slides' meanimages")
        p.add_argument('--flatfield-model-file',type=pathlib.Path,
                        default=CONST.DEFAULT_FLATFIELD_MODEL_FILEPATH,
                        help='path to a .csv file defining which slides should be used for the given version')
        p.add_argument('--meanimage_dirname',default=UNIV_CONST.MEANIMAGE_DIRNAME,
                       help='''The name of the meanimage directories to use 
                                (useful if multiple meanimage versions have been created)''')
        p.add_argument('--outdir',type=pathlib.Path,default=CONST.DEFAULT_FLATFIELD_MODEL_DIR.parent,
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
            all_model_table_entries = readtable(flatfield_model_file,FlatfieldModelTableEntry)
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

class BatchFlatfieldMultiCohortComponentTiff(BatchFlatfieldMultiCohortBase) :
    singlecohortclass = BatchFlatfieldCohortComponentTiff
    flatfield_type = FlatfieldComponentTiff
class BatchFlatfieldMultiCohortIm3(BatchFlatfieldMultiCohortBase) :
    singlecohortclass = BatchFlatfieldCohortIm3
    flatfield_type = FlatfieldIm3

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    BatchFlatfieldMultiCohortIm3.runfromargumentparser(args)

if __name__=='__main__' :
    main()
