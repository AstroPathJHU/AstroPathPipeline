#imports 
from .warp_fitter import WarpFitter
from .utilities import warp_logger, addCommonWarpingArgumentsToParser, checkDirAndFixedArgs, findSampleOctets, readOctetsFromFile
from .config import CONST
from ..utilities.misc import split_csv_to_list, split_csv_to_list_of_ints, split_csv_to_dict_of_floats, split_csv_to_dict_of_bounds
from argparse import ArgumentParser
import multiprocessing as mp
import os, gc
import cProfile

#################### FILE-SCOPE CONSTANTS ####################

#placeholder default arguments for command line options
DEFAULT_OVERLAPS    = '-999' 
DEFAULT_OCTETS      = '-999' 

#################### HELPER FUNCTIONS ####################

#helper function to check all command-line arguments
def checkArgs(args) :
    #make sure directories exist and fixed parameter choice is valid
    checkDirAndFixedArgs(args)
    #split some of the other arguments
    args.normalize = split_csv_to_list(args.normalize)
    args.init_pars = CONST.DEFAULT_INIT_PARS if args.init_pars is None else split_csv_to_dict_of_floats(args.init_pars)
    args.init_bounds = CONST.DEFAULT_INIT_BOUNDS if args.init_bounds is None else split_csv_to_dict_of_bounds(args.init_bounds)
    #only one of "overlaps" and/or "octets" can be specified
    nspec = sum([args.overlaps!=split_csv_to_list_of_ints(DEFAULT_OVERLAPS),
                 args.octets!=split_csv_to_list_of_ints(DEFAULT_OCTETS)])
    if nspec==0 :
        warp_logger.info('No overlaps or octets specified; will use the default of octets=[-1] to run all valid octets.')
        args.octets=[-1]
    nspec = sum([args.overlaps!=split_csv_to_list_of_ints(DEFAULT_OVERLAPS),
                 args.octets!=split_csv_to_list_of_ints(DEFAULT_OCTETS)])
    if nspec!=1 :
        raise ValueError(f'Must specify exactly ONE of overlaps or octets! (overlaps={args.overlaps}, octets={args.octets})')
    #the threshold file must exist if it's to be used
    if args.threshold_file_dir is not None :
        if not os.path.isdir(args.threshold_file_dir) :
            raise ValueError(f'ERROR: threshold_file_dir ({args.threshold_file_dir}) does not exist!')
        tfp = os.path.join(args.threshold_file_dir,f'{args.sample}{CONST.THRESHOLD_FILE_EXT}')
        if not os.path.isfile(tfp) :
            raise ValueError(f'ERROR: threshold_file_dir does not contain a threshold file for this sample ({tfp})!')
    #The user must specify either an octet run dir or a threshold file dir
    if args.threshold_file_dir is None and args.octet_run_dir is None :
        raise ValueError('ERROR: must specify either an octet_run_dir or a threshold_file_dir!')
    #if the thresholding file dir and the octet dir are both provided the user needs to disambiguate
    if args.threshold_file_dir is not None and args.octet_run_dir is not None :
        raise ValueError('ERROR: cannot specify both an octet_run_dir and a threshold_file_dir!')

# Helper function to determine the list of overlaps
def getOverlaps(args) :
    #the threshold file must exist if it's to be used
    if args.threshold_file_dir is not None :
        if not os.path.isdir(args.threshold_file_dir) :
            raise ValueError(f'ERROR: threshold_file_dir ({args.threshold_file_dir}) does not exist!')
        tfp = os.path.join(args.threshold_file_dir,f'{args.sample}{CONST.THRESHOLD_FILE_EXT}')
        if not os.path.isfile(tfp) :
            raise ValueError(f'ERROR: threshold_file_dir does not contain a threshold file for this sample ({tfp})!')
    #if the thresholding file dir and the octet dir are both provided the user needs to disambiguate
    if args.threshold_file_dir is not None and args.octet_run_dir is not None :
        raise ValueError('ERROR: cannot specify both an octet_run_dir and a threshold_file_dir!')
    #set the overlaps variable based on which of the options was used to specify
    overlaps=[]
    #if the overlaps are being specified then they have to be either -1 (to use all), a tuple (to use a range), or a list
    if args.overlaps!=split_csv_to_list_of_ints(DEFAULT_OVERLAPS) :
        overlaps = args.overlaps
        if overlaps==[-1] :
            overlaps = overlaps[0]
        elif len(overlaps)==2 :
            overlaps=tuple(overlaps)
        return overlaps
    #otherwise overlaps will have to be set after finding the octets
    else :
        #read in the octets if they have already been defined for this sample
        octet_run_dir = args.octet_run_dir if args.octet_run_dir is not None else args.workingdir_name
        if os.path.isfile(os.path.join(octet_run_dir,f'{args.sample}{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM}')) :
            valid_octets = readOctetsFromFile(octet_run_dir,args.rawfile_top_dir,args.metadata_top_dir,args.sample,args.layer)
        #otherwise run an alignment to find the valid octets get the dictionary of overlap octets
        else :
            threshold_file_path=os.path.join(args.threshold_file_dir,f'{args.sample}{CONST.THRESHOLD_FILE_EXT}')
            valid_octets = findSampleOctets(args.rawfile_top_dir,args.metadata_top_dir,threshold_file_path,args.req_pixel_frac,args.sample,
                                            args.workingdir_name,args.n_threads,args.layer)
        if args.mode in ('fit', 'check_run', 'cProfile') and args.octets!=split_csv_to_list_of_ints(DEFAULT_OCTETS):
            for i,octet in enumerate(valid_octets,start=1) :
                if i in args.octets or args.octets==[-1]:
                    warp_logger.info(f'Adding overlaps in octet #{i}...')
                    overlaps+=octet.overlap_ns
            if (args.octets!=[-1] and len(overlaps)!=8*len(args.octets)) or (args.octets==[-1] and len(overlaps)!=8*len(valid_octets)) :
                msg =f'specified octets {args.octets} did not result in the desired set of overlaps! '
                msg+=f'(asked for {len(args.octets)} octets but found {len(overlaps)} corresponding overlaps)'
                msg+=f' there are {len(valid_octets)} octets to choose from.'
                raise ValueError(msg)
    return overlaps

#################### MAIN SCRIPT ####################

if __name__=='__main__' :
    mp.freeze_support()
    #define and get the command-line arguments
    parser = ArgumentParser()
    #positional arguments
    parser.add_argument('mode',             help='Operation to perform', choices=['fit','find_octets','check_run','cProfile'])
    #add the common arguments
    addCommonWarpingArgumentsToParser(parser)
    #additional group for how to figure out which overlaps will be used
    overlap_selection_group = parser.add_argument_group('overlap selection', 'what set of overlaps should be used?')
    overlap_selection_group.add_argument('--octet_run_dir', 
                                         help=f'Path to a previously-created workingdir that contains a [sample]_{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM} file')
    overlap_selection_group.add_argument('--overlaps',       default=DEFAULT_OVERLAPS, type=split_csv_to_list_of_ints,         
                                         help='Comma-separated list of numbers (n) of the overlaps to use (two-element defines a range)')
    overlap_selection_group.add_argument('--octets',         default=DEFAULT_OCTETS,   type=split_csv_to_list_of_ints,         
                                         help='Comma-separated list of overlap octet indices (ordered by n of octet central rectangle) to use')
    #how many threads to use in finding octets/reading files
    parser.add_argument('--n_threads',      default=10, type=int,         
                        help='Maximum number of threads/processes to run at once')
    args = parser.parse_args()
    #apply some checks to the arguments to make sure they're valid
    checkArgs(args)
    #get the overlaps to use for fitting
    overlaps=getOverlaps(args)
    gc.collect()
    #setup and run
    if args.mode in ('fit', 'check_run', 'cProfile') :
        #how many overlaps will be used
        warp_logger.info(f'Will run fit on a sample of {len(overlaps)} total overlaps.')
        #make the WarpFitter Objects
        warp_logger.info('Initializing WarpFitter')
        fitter = WarpFitter(args.sample,args.rawfile_top_dir,args.metadata_top_dir,args.workingdir_name,overlaps,args.layer)
        #figure out which parameters will be fixed
        fix_cxcy   = 'cx' in args.fixed and 'cy' in args.fixed
        fix_fxfy   = 'fx' in args.fixed and 'fy' in args.fixed
        fix_k1k2k3 = 'k1' in args.fixed and 'k2' in args.fixed and 'k3' in args.fixed
        fix_p1p2   = 'p1' in args.fixed and 'p2' in args.fixed
        #check the run if that's what's being asked
        if args.mode in ('check_run') :
            fitter.checkFit(fixed=args.fixed,normalize=args.normalize,init_pars=args.init_pars,init_bounds=args.init_bounds,
                            float_p1p2_in_polish_fit=args.float_p1p2_to_polish,
                            max_radial_warp=args.max_radial_warp,max_tangential_warp=args.max_tangential_warp,
                            p1p2_polish_lasso_lambda=args.p1p2_polish_lasso_lambda,polish=True)
        #otherwise actually run it
        elif args.mode in ('fit', 'cProfile') :
            #load the raw files
            warp_logger.info('Loading raw files')
            fitter.loadRawFiles(args.flatfield_file,args.exposure_time_offset_file,args.n_threads)
             #fit the model to the data
            warp_logger.info('Running doFit')
            if args.mode == 'fit' :
                fitter.doFit(fixed=args.fixed,normalize=args.normalize,init_pars=args.init_pars,init_bounds=args.init_bounds,
                             float_p1p2_in_polish_fit=args.float_p1p2_to_polish,
                             max_radial_warp=args.max_radial_warp,max_tangential_warp=args.max_tangential_warp,
                             p1p2_polish_lasso_lambda=args.p1p2_polish_lasso_lambda,polish=True,
                             print_every=args.print_every,maxiter=args.max_iter)
            elif args.mode == 'cProfile' :
                cProfile.run("""fitter.doFit(fixed=args.fixed,normalize=args.normalize,init_pars=args.init_pars,init_bounds=args.init_bounds,
                                float_p1p2_in_polish_fit=args.float_p1p2_to_polish,
                                max_radial_warp=args.max_radial_warp,max_tangential_warp=args.max_tangential_warp,
                                p1p2_polish_lasso_lambda=args.p1p2_polish_lasso_lambda,polish=True,
                                print_every=args.print_every,maxiter=args.max_iter)""")

    warp_logger.info('All done : )')

