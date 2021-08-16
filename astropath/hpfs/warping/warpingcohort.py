#imports
import sys, traceback
import multiprocessing as mp
from random import choices
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.misc import get_GPU_thread
from ...utilities.tableio import writetable, readtable
from ...shared.argumentparser import WarpFitArgumentParser, WorkingDirArgumentParser, FileTypeArgumentParser
from ...shared.argumentparser import GPUArgumentParser
from ...shared.cohort import CorrectedImageCohort, SelectLayersCohort, WorkflowCohort
from .config import CONST
from .utilities import OverlapOctet, WarpFitResult
from .plotting import principal_point_plot, rad_warp_amt_plots, rad_warp_par_plots, warp_field_variation_plots
from .latexsummary import FitGroupLatexSummary
from .warpingsample import WarpingSample
from .warpfit import WarpFit


class WarpingCohort(CorrectedImageCohort,SelectLayersCohort,WorkflowCohort,WarpFitArgumentParser,
                    WorkingDirArgumentParser,GPUArgumentParser) :
    """
    Class to perform a set of warping fits for a cohort
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,layer,fit_1_octets,fit_2_octets,fit_3_octets,fit_1_iters,fit_2_iters,fit_3_iters,
                 fixed,init_pars,bounds,max_rad_warp,max_tan_warp,workingdir=None,useGPU=True,**kwargs) :
        super().__init__(*args,**kwargs)
        #set variables for how the fits should run
        self.__layer = layer
        self.__n_fit_1_octets = fit_1_octets
        self.__n_fit_2_octets = fit_2_octets
        self.__n_fit_3_octets = fit_3_octets
        self.__fit_1_iters = fit_1_iters
        self.__fit_2_iters = fit_2_iters
        self.__fit_3_iters = fit_3_iters
        self.__fixed = fixed
        #next two parameters below are one-element lists because they are parsed as dictionaries
        self.__init_pars = init_pars[0]
        self.__init_bounds = bounds[0]
        self.__max_rad_warp = max_rad_warp
        self.__max_tan_warp = max_tan_warp
        #if the working directory wasn't given, set it to the "Warping" directory inside the root directory
        self.__workingdir = workingdir
        if self.__workingdir is None :
            self.__workingdir = self.root / UNIV_CONST.WARPING_DIRNAME
        if not self.__workingdir.is_dir() :
            self.__workingdir.mkdir(parents=True)
        self.__useGPU = useGPU
        #if running with the GPU, create a GPU thread and start a dictionary of GPU FFTs to give to each sample
        self.gputhread = get_GPU_thread(sys.platform=='darwin') if useGPU else None
        self.gpufftdict = None if self.gputhread is None else {}
        #placeholders for the groups of octets
        self.__fit_1_octets = []; self.__fit_2_octets = []; self.__fit_3_octets = []

    def run(self,**kwargs) :
        # First check to see if the octet groups have already been defined
        self.__get_octets_if_existing()
        # If the above didn't populate the dictionaries then we have to find the octets to use from the samples
        if ( self.__fit_1_octets==[] and self.__fit_2_octets==[] and 
             self.__fit_3_octets==[] ) :
            #make a list of all octets
            self.__octets = []
            # Run all of the individual samples first (runs octet finding, which is independent for every sample)
            super().run(**kwargs)
            # Randomly separate the octets into the three fit groups of the requested size
            self.__split_octets()
        # Run the three fit groups
        with self.globallogger() as logger :
            self.__run_initial_pattern_fits(logger)

    def runsample(self,sample,**kwargs) :
        #actually run the sample
        super().runsample(sample,**kwargs)
        #add the sample's octets to the overall dictionary
        self.__octets += sample.octets

    #################### CLASS VARIABLES + PROPERTIES ####################

    sampleclass = WarpingSample

    @property
    def workingdir(self) :
        return self.__workingdir
    @property
    def fit_1_octet_fp(self) :
        return self.__workingdir / CONST.OCTET_SUBDIR_NAME / 'initial_pattern_octets_selected.csv'
    @property
    def fit_1_results_fp(self) :
        return self.__workingdir / f'initial_pattern_fit_results_{self.__n_fit_1_octets}_layer_{self.__layer}.csv'
    @property
    def fit_2_octet_fp(self) :
        return self.__workingdir / CONST.OCTET_SUBDIR_NAME / 'principal_point_octets_selected.csv'
    @property
    def fit_2_results_fp(self) :
        return self.__workingdir / f'principal_point_fit_results_{self.__n_fit_2_octets}_layer_{self.__layer}.csv'
    @property
    def fit_3_octet_fp(self) :
        return self.__workingdir / CONST.OCTET_SUBDIR_NAME / 'final_pattern_octets_selected.csv'
    @property
    def fit_3_results_fp(self) :
        return self.__workingdir / f'final_pattern_fit_results_{self.__n_fit_3_octets}_layer_{self.__layer}.csv'
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

    #################### PRIVATE HELPER METHODS ####################

    def __get_octets_if_existing(self) :
        """
        If files listing the octets for a sample have been written out, read them into the dictionaries for the cohort
        """
        if self.fit_1_octet_fp.is_file() and self.fit_2_octet_fp.is_file() and self.fit_3_octet_fp.is_file() :
            fit_1_octets = readtable(self.fit_1_octet_fp,OverlapOctet)
            fit_2_octets = readtable(self.fit_2_octet_fp,OverlapOctet)
            fit_3_octets = readtable(self.fit_3_octet_fp,OverlapOctet)
            if ( len(fit_1_octets)==self.__n_fit_1_octets and len(fit_2_octets)==self.__n_fit_2_octets and 
                len(fit_3_octets)==self.__n_fit_3_octets ) :
                msg = f'Will use previously chosen octets listed in {self.fit_1_octet_fp}, {self.fit_2_octet_fp}, '
                msg+= f'and {self.fit_3_octet_fp}'
                self.logger.info(msg)
                self.__fit_1_octets = fit_1_octets
                self.__fit_2_octets = fit_2_octets
                self.__fit_3_octets = fit_3_octets

    def __split_octets(self) :
        """
        Separate all of the octets found into three subgroups for each set of fits
        """
        #make sure there are enough octets overall 
        n_total_octets_needed = self.__n_fit_1_octets+self.__n_fit_2_octets+self.__n_fit_3_octets
        n_octets_available = len(self.__octets)
        if n_octets_available<n_total_octets_needed :
            errmsg = f'ERROR: only found {n_octets_available} octets in the cohort overall, but '
            errmsg+= f'{n_total_octets_needed} are needed to run all three sets of fit groups! '
            errmsg+= 'Please request fewer octets to use in fitting.'
            raise RuntimeError(errmsg)
        #randomly choose the three subsets of octets
        selected_octets = choices(self.__octets,k=n_total_octets_needed)
        self.__fit_1_octets = selected_octets[:self.__n_fit_1_octets]
        self.__fit_2_octets = selected_octets[self.__n_fit_1_octets:self.__n_fit_1_octets+self.__n_fit_2_octets]
        self.__fit_3_octets = selected_octets[-self.__n_fit_3_octets:]
        #write out files listing the octets
        writetable(self.fit_1_octet_fp,self.__fit_1_octets)
        writetable(self.fit_2_octet_fp,self.__fit_2_octets)
        writetable(self.fit_3_octet_fp,self.__fit_3_octets)

    def __run_initial_pattern_fits(self,logger) :
        """
        Run fits for the first chosen group of octets and write out all of the individual results.
        Set a variable to use to initialize the second group of principal point fits.
        If the fit result file already exists, just read it and set the result needed.
        """
        #first check to see if the output file exists; if it does then just read the results from it
        if self.fit_1_results_fp.is_file() :
            fit_1_results = readtable(self.fit_1_results_fp,WarpFitResult)
        else :
            #if it doesn't exist yet though we need to run each of the fits and then write it out
            fit_1_results = []; fit_1_field_logs = []; fit_1_metadata_summaries = []
            for oi,o in enumerate(self.__fit_1_octets) :
                msg = f'Running initial pattern fit for octet around {o.slide_ID} rectangle {o.p1_rect_n} '
                msg+= f'({oi+1} of {len(self.__fit_1_octets)})....'
                self.logger.debug(msg)
                this_sample = None
                for s in self.samples :
                    if s.SlideID==o.slide_ID :
                        this_sample = s
                        break
                if this_sample is None :
                    errmsg = f'ERROR: unable to find the appropriate initialized sample for slide {o.slide_ID}'
                    raise RuntimeError(errmsg)
                warp_fit = WarpFit(this_sample,o,logger)
                try :
                    wfr,wffls,wfms = warp_fit.run(self.__fit_1_iters,self.__fixed,
                                                  self.__init_pars,self.__init_bounds,
                                                  self.__max_rad_warp,
                                                  self.__max_tan_warp)
                    fit_1_results.append(wfr)
                    fit_1_field_logs+=wffls
                    fit_1_metadata_summaries.append(wfms)
                except Exception as e :
                    warnmsg = f'WARNING: initial pattern fit for octet around {o.slide_ID} rectangle {o.p1_rect_n} '
                    warnmsg+= f'failed with the error "{e}" and this result will be ignored moving forward. More info '
                    warnmsg+= 'on the error below.'
                    self.logger.warning(warnmsg)
                    try :
                        raise e
                    except Exception :
                        for l in traceback.format_exc().split('\n') :
                            self.logger.info(l)  
            if len(fit_1_field_logs)>0 :
                writetable(self.__workingdir / 'initial_pattern_fit_field_logs.csv',fit_1_field_logs)
            if len(fit_1_metadata_summaries)>0 :
                writetable(self.__workingdir / 'initial_pattern_fit_metadata_summaries.csv',fit_1_metadata_summaries)
        #collect the results together
        if len(fit_1_results)<1 :
            raise RuntimeError(f'ERROR: {len(fit_1_results)} fit results were obtained for the initial pattern!')
        good_results = [r for r in fit_1_results if r.cost_reduction>0]
        if len(good_results)<1 :
            raise RuntimeError(f'ERROR: {len(good_results)} fits for the initial pattern resulted in reduced cost!')
        #write out some plots and collect them in a summary .pdf
        try :
            plot_name_stem = f'initial_pattern_fits'
            plotdirname = 'initial_pattern_fit_plots'
            savedir = self.__workingdir / plotdirname
            if not savedir.is_dir() :
                savedir.mkdir(parents=True)
            principal_point_plot(fit_1_results,save_stem=plot_name_stem,save_dir=savedir)
            rad_warp_amt_plots(fit_1_results,save_stem=plot_name_stem,save_dir=savedir)
            rad_warp_par_plots(fit_1_results,save_stem=plot_name_stem,save_dir=savedir)
            fit_iteration_plot(fit_1_results,save_stem=plot_name_stem,save_dir=savedir)
            warp_field_variation_plots(fit_1_results,save_stem=plot_name_stem,save_dir=savedir)
        except Exception as e :
            warp_logger.warning(f'WARNING: failed to create plots for group of results. Exception: {e}')
        self.logger.info('Making the summary pdf....')
        latex_summary = FitGroupLatexSummary(savedir,plot_name_stem,'Initial Pattern Fit Group')
        latex_summary.build_tex_file()
        check = latex_summary.compile()
        if check!=0 :
            warnmsg = 'WARNING: failed while compiling fit group summary LaTeX file into a PDF. '
            warnmsg+= f'tex file will be in {latex_summary.failed_compilation_tex_file_path}'
            self.logger.warning(warnmsg)
        #set variables to define and constrain the next group of fits
        self.__fit_2_init_pars = {}
        sum_weights = np.sum(np.array([r.cost_reduction for r in good_results]))
        for fpn in CONST.ORDERED_FIT_PAR_NAMES :
            if fpn=='cx' or fpn=='cy' :
                if fpn in self.__init_pars.keys() :
                    self.__fit_2_init_pars[fpn] = self.__init_pars[fpn]
                continue
            elif fpn=='fx' :
                weighted_sum = np.sum(np.array([r.fx*r.cost_reduction for r in good_results]))
            elif fpn=='fy' :
                weighted_sum = np.sum(np.array([r.fy*r.cost_reduction for r in good_results]))
            elif fpn=='k1' :
                weighted_sum = np.sum(np.array([r.k1*r.cost_reduction for r in good_results]))
            elif fpn=='k2' :
                weighted_sum = np.sum(np.array([r.k2*r.cost_reduction for r in good_results]))
            elif fpn=='k3' :
                weighted_sum = np.sum(np.array([r.k3*r.cost_reduction for r in good_results]))
            elif fpn=='p1' :
                weighted_sum = np.sum(np.array([r.p1*r.cost_reduction for r in good_results]))
            elif fpn=='p2' :
                weighted_sum = np.sum(np.array([r.p2*r.cost_reduction for r in good_results]))
            self.__fit_2_init_pars[fpn] = weighted_sum/sum_weights

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    WarpingCohort.runfromargumentparser(args)

if __name__=='__main__' :
    main()