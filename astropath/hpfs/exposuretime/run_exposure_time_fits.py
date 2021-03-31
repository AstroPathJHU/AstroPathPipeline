#imports
from .exposure_time_fit_group import ExposureTimeOffsetFitGroup
from .utilities import et_fit_logger, checkArgs
from ...utilities.misc import split_csv_to_list_of_ints, addCommonArgumentsToParser
from argparse import ArgumentParser
import multiprocessing as mp

#################### MAIN SCRIPT ####################

def main(args=None) :
    mp.freeze_support()
    if args is None :
        #define and get the command-line arguments
        parser = ArgumentParser()
        #add the common arguments, just the positional and flatfielding ones
        addCommonArgumentsToParser(parser,et_correction=False,warping=False)
        #group for options of how the images whould be processed
        image_processing_group = parser.add_argument_group('fit options', 'how should the fit be done?')
        image_processing_group.add_argument('--smooth_sigma',         default=3., type=float,
                                            help='sigma (in pixels) for initial Gaussian blur of images')
        image_processing_group.add_argument('--use_whole_image', action='store_true',
                                            help='Add this flag to use the entire images rather than the outer 50%% of each overlap')
        #group for options of how the fits will proceed
        fit_option_group = parser.add_argument_group('fit options', 'how should the fits be done?')
        fit_option_group.add_argument('--initial_offset', default=5.,      type=float,
                                      help='Dark current offset count fit starting point (default=50.)')
        fit_option_group.add_argument('--min_pixel_frac_for_offset_limit',  default=1e-4, type=float,         
                                      help="""Upper limit of offsets to search will be increased until at least this fraction of pixels in some image 
                                              are less than that value (prevents the upper limit being an outlier single pixel).""")
        fit_option_group.add_argument('--max_iter',       default=15000,    type=int,
                                      help='Maximum number of fit iterations (default=15000)')
        fit_option_group.add_argument('--gtol',           default=1e-8,     type=float,
                                      help='Minimization stops when the projected gradient is less than this (default=1e-7).')
        fit_option_group.add_argument('--eps',            default=0.25,        type=float,
                                      help='Step size around current value for Jacobian approximation (default=0.25).')
        fit_option_group.add_argument('--print_every',    default=10,       type=int,
                                      help='How many iterations to wait between printing minimization progress (default=10)')
        #group for other run options
        run_option_group = parser.add_argument_group('run options', 'other options for this run')
        run_option_group.add_argument('--n_threads',             default=5,   type=int,         
                                      help='Maximum number of threads/processes to run at once (different layers run in parallel).')
        run_option_group.add_argument('--layers',                default=[-1], type=split_csv_to_list_of_ints,         
                                      help='CSV list of image layer numbers to use (indexed from 1) [default=-1 runs all layers]')
        run_option_group.add_argument('--overlaps',              default=[-1], type=split_csv_to_list_of_ints,         
                                      help='CSV list of overlap numbers to use [default=-1 runs all of them]. Should really only use this for testing.')
        run_option_group.add_argument('--n_comparisons_to_save', default=15,   type=int,         
                                      help='Number of pre/post-fit overlap overlay comparison images to save (default=15)')
        run_option_group.add_argument('--allow_edge_HPFs', action='store_true',
                                      help="""Add this flag to allow overlaps with HPFs on the tissue edges""")
        args = parser.parse_args(args=args)
    #make sure the arguments are valid
    checkArgs(args)
    #initialize a fit
    et_fit_logger.info('Defining group of fits....')
    fit_group = ExposureTimeOffsetFitGroup(args.slideID,args.rawfile_top_dir,args.root_dir,args.workingdir,args.layers,args.n_threads)
    #run the fits
    et_fit_logger.info('Running fits....')
    fit_group.runFits(args.flatfield_file,args.overlaps,args.smooth_sigma,args.use_whole_image,
                      args.initial_offset,args.min_pixel_frac_for_offset_limit,args.max_iter,args.gtol,args.eps,args.print_every,
                      args.n_comparisons_to_save,args.allow_edge_HPFs)
    et_fit_logger.info('Done!')

if __name__=='__main__' :
    main()
