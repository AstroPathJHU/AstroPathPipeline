#imports
from .exposure_time_fit_group import ExposureTimeOffsetFitGroup
from .utilities import et_fit_logger, checkArgs
from ..utilities.misc import split_csv_to_list_of_ints
from argparse import ArgumentParser
import multiprocessing as mp

#################### MAIN SCRIPT ####################

if __name__=='__main__' :
    mp.freeze_support()
    #define and get the command-line arguments
    parser = ArgumentParser()
    #positional arguments
    parser.add_argument('sample',           help='Name of the data sample to use')
    parser.add_argument('rawfile_top_dir',  help='Path to the directory containing the "[sample_name]/*.Data.dat" files')
    parser.add_argument('metadata_top_dir', help='Path to the directory containing "[sample name]/im3/xml" subdirectories')
    parser.add_argument('flatfield_file',   help='Path to the flatfield.bin file that should be applied to files in this sample')
    parser.add_argument('workingdir_name',  help='Name of the working directory that will be created')
    #group for options of how the images whould be processed
    image_processing_group = parser.add_argument_group('fit options', 'how should the fit be done?')
    image_processing_group.add_argument('--smooth_sigma',         default=3., type=float,
                                        help='Minimization stops when the projected gradient is less than this')
    image_processing_group.add_argument('--central_regions_only', action='store_true',
                                        help='Add this flag to cut out the outer 50%% of each overlap')
    #group for options of how the fits will proceed
    fit_option_group = parser.add_argument_group('fit options', 'how should the fits be done?')
    fit_option_group.add_argument('--initial_offset', default=50.,      type=float,
                                  help='Dark current offset count fit starting point (default=50.)')
    fit_option_group.add_argument('--offset_bounds',  default=[0,1000], type=split_csv_to_list_of_ints,         
                                  help='CSV of low,high bounds for offset [default=(0,1000)]')
    fit_option_group.add_argument('--max_iter',       default=15000,    type=int,
                                  help='Maximum number of fit iterations (default=15000)')
    fit_option_group.add_argument('--gtol',           default=1e-7,     type=float,
                                  help='Minimization stops when the projected gradient is less than this (default=1e-7).')
    fit_option_group.add_argument('--eps',            default=1,        type=float,
                                  help='Step size around current value for Jacobian approximation (default=1).')
    fit_option_group.add_argument('--print_every',    default=10,       type=int,
                                  help='How many iterations to wait between printing minimization progress (default=10)')
    #group for other run options
    run_option_group = parser.add_argument_group('run options', 'other options for this run')
    run_option_group.add_argument('--n_threads',             default=10,   type=int,         
                                  help='Maximum number of threads/processes to run at once (different layers run in parallel).')
    run_option_group.add_argument('--layers',                default=[-1], type=split_csv_to_list_of_ints,         
                                  help='CSV list of image layer numbers to use (indexed from 1) [default=-1 runs all layers]')
    run_option_group.add_argument('--overlaps',              default=[-1], type=split_csv_to_list_of_ints,         
                                  help='CSV list of overlap numbers to use [default=-1 runs all of them]. Should really only use this for testing.')
    run_option_group.add_argument('--n_comparisons_to_save', default=30,   type=int,         
                                  help='Number of pre/post-fit overlap overlay comparison images to save (default=30)')
    args = parser.parse_args()
    #make sure the arguments are valid
    checkArgs(args)
    #initialize a fit
    et_fit_logger.info('Defining group of fits....')
    fit_group = ExposureTimeOffsetFitGroup(args.sample,args.rawfile_top_dir,args.metadata_top_dir,args.workingdir_name,args.layers,args.n_threads)
    #prepare the fits to run by reading the files and exposure times etc.
    et_fit_logger.info('Loading files and preparing fits....')
    fit_group.prepFits(args.flatfield_file,args.overlaps,args.smooth_sigma,args.central_regions_only)
    #run the fits
    et_fit_logger.info('Running fits....')
    fit_group.runFits(args.initial_offset,args.offset_bounds,args.max_iter,args.gtol,args.eps,args.print_every)
    #write out the fit results
    et_fit_logger.info('Writing out fit results....')
    fit_group.writeOutResults(args.n_comparisons_to_save)
    et_fit_logger.info('Done!')
