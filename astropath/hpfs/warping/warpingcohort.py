#imports
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.misc import split_csv_to_dict_of_floats, split_csv_to_dict_of_bounds
from ...shared.argumentparser import WarpFitArgumentParser, WorkingDirArgumentParser, FileTypeArgumentParser
from ...shared.argumentparser import GPUArgumentParser, ParallelArgumentParser
from ...shared.cohort import CorrectedImageCohort, SelectLayersCohort, WorkflowCohort
from .warpingsample import WarpingSample


class WarpingCohort(CorrectedImageCohort,SelectLayersCohort,WorkflowCohort,WarpFitArgumentParser,
                    WorkingDirArgumentParser,GPUArgumentParser,ParallelArgumentParser) :
    """
    Class to perform a set of warping fits for a cohort
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,layer,fit_1_octets,fit_2_octets,fit_3_octets,fit_1_iters,fit_2_iters,fit_3_iters,
                 fixed,init_pars,bounds,max_rad_warp,max_tan_warp,workingdir=None,useGPU=True,njobs=5,**kwargs) :
        super().__init__(*args,**kwargs)
        #set variables for how the fits should run
        self.__layer = layer
        self.__fit_1_octets = fit_1_octets
        self.__fit_2_octets = fit_2_octets
        self.__fit_3_octets = fit_3_octets
        self.__fit_1_iters = fit_1_iters
        self.__fit_2_iters = fit_2_iters
        self.__fit_3_iters = fit_3_iters
        self.__fixed = fixed
        self.__init_pars = init_pars
        self.__init_bounds = bounds
        self.__max_rad_warp = max_rad_warp
        self.__max_tan_warp = max_tan_warp
        #if the working directory wasn't given, set it to the "Warping" directory inside the root directory
        self.__workingdir = workingdir
        if self.__workingdir is None :
            self.__workingdir = self.root / UNIV_CONST.WARPING_DIRNAME
        if not self.__workingdir.is_dir() :
            self.__workingdir.mkdir(parents=True)
        self.__useGPU = useGPU
        self.__njobs = njobs
        #placeholders for metadata collection
        self.__field_logs = []
        self.__metadata_summaries = []

    def run(self,**kwargs) :
        # Run all of the individual samples first (runs octet finding, which is independent for every sample)
        super().run(**kwargs)
        # Get every sample's octets and randomly separate them into the three fit groups

    #################### CLASS VARIABLES + PROPERTIES ####################

    sampleclass = WarpingSample

    @property
    def workingdir(self) :
        return self.__workingdir

    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        p.add_argument('--layer', type=int, default=1,
                       help='The layer number (starting from one) of the images that should be used (default=1)')
        p.add_argument('--initial_pattern_octets', type=int, default=50,
                       help='Number of octets to use in the initial pattern fits (default=50)')
        p.add_argument('--principal_point_octets', type=int, default=50,
                       help='Number of octets to use in the principal point location fits (default=50)')
        p.add_argument('--final_pattern_octets',   type=int, default=100,
                       help='Number of octets to use in the final pattern fits (default=100)')
        p.add_argument('--initial_pattern_max_iters', type=int, default=200,
                       help='Max # of iterations to run in the initial pattern fits (default=200)')
        p.add_argument('--principal_point_max_iters', type=int, default=500,
                       help='Max # of iterations to run in the principal point location fits (default=500)')
        p.add_argument('--final_pattern_max_iters',   type=int, default=500,
                       help='Max # of iterations to run in the final pattern fits (default=500)')
        return p
    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        return {
            **super().initkwargsfromargumentparser(parsed_args_dict),
            'layer': parsed_args_dict.pop('layer'),
            'fit_1_octets': parsed_args_dict.pop('initial_pattern_octets'),
            'fit_2_octets': parsed_args_dict.pop('principal_point_octets'),
            'fit_3_octets': parsed_args_dict.pop('final_pattern_octets'),
            'fit_1_iters': parsed_args_dict.pop('initial_pattern_max_iters'),
            'fit_2_iters': parsed_args_dict.pop('principal_point_max_iters'),
            'fit_3_iters': parsed_args_dict.pop('final_pattern_max_iters'),
        }
    @property
    def initiatesamplekwargs(self) :
        return {**super().initiatesamplekwargs,
                'filetype':'raw',
                'layer': self.__layer,
                'workingdir':self.__workingdir,
                'useGPU':self.__useGPU,
               }
    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':False}

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    WarpingCohort.runfromargumentparser(args)

if __name__=='__main__' :
    main()