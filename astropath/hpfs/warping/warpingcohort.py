#imports
import sys
from random import choices
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.misc import get_GPU_thread
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
        #if running with the GPU, create a GPU thread and start a dictionary of GPU FFTs to give to each sample
        self.gputhread = get_GPU_thread(sys.platform=='darwin') if useGPU else None
        self.gpufftdict = None if self.gputhread is None else {}
        #make a dictionary of all octets
        self.__octets = {}
        #placeholders for metadata collection
        self.__field_logs = []
        self.__metadata_summaries = []

    def run(self,**kwargs) :
        # Run all of the individual samples first (runs octet finding, which is independent for every sample)
        super().run(**kwargs)
        # Randomly separate the octets into the three fit groups of the requested size
        self.__split_octets()
        # Run the three fit groups

    def runsample(self,sample,**kwargs) :
        #actually run the sample
        super().runsample(sample,**kwargs)
        #add the sample's octets to the overall dictionary
        print(f'octets for {sample.SlideID} = {sample.octets}')
        self.__octets[sample.SlideID] = sample.octets

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
        parsed_args_dict['skip_finished']=False #rerun every sample. If their output exists they'll just pick it up.
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
                'gputhread':self.gputhread,
                'gpufftdict':self.gpufftdict,
               }
    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':False,'workingdir':self.__workingdir}

    #################### PRIVATE HELPER METHODS ####################

    def __split_octets(self) :
        """
        Separate all of the octets found into three subgroups for each set of fits
        """
        #make sure there are enough octets overall 
        n_total_octets_needed = self.__fit_1_octets+self.__fit_2_octets+self.__fit_3_octets
        if len(self.__octets)<n_total_octets_needed :
            errmsg = f'ERROR: only found {len(self.__octets)} octets in the cohort overall, but '
            errmsg+= f'{n_total_octets_needed} are needed to run all three sets of fit groups! '
            errmsg+= 'Please request fewer octets to use in fitting.'
            raise RuntimeError(errmsg)
        #randomly choose the three subsets of octets
        octet_tuples = list(self.__octets.items())
        selected_octet_tuples = choices(octet_tuples,n_total_octets_needed)
        self.__fit_1_octets_by_sample = {}; self.__fit_2_octets_by_sample = {}; self.__fit_3_octets_by_sample = {}
        for ot in selected_octet_tuples[:self.__fit_1_octets] :
            if ot[0] not in self.__fit_1_octets_by_sample.keys() :
                self.__fit_1_octets_by_sample[ot[0]] = []
            self.__fit_1_octets_by_sample[ot[0]].append(ot[1])
        for ot in selected_octet_tuples[self.__fit_1_octets:self.__fit_1_octets+self.__fit_2_octets] :
            if ot[0] not in self.__fit_2_octets_by_sample.keys() :
                self.__fit_2_octets_by_sample[ot[0]] = []
            self.__fit_2_octets_by_sample[ot[0]].append(ot[1])
        for ot in selected_octet_tuples[-self.__fit_3_octets:] :
            if ot[0] not in self.__fit_3_octets_by_sample.keys() :
                self.__fit_3_octets_by_sample[ot[0]] = []
            self.__fit_3_octets_by_sample[ot[0]].append(ot[1])

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    WarpingCohort.runfromargumentparser(args)

if __name__=='__main__' :
    main()