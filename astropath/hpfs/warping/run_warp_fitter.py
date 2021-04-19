#imports 
from .warp_fitter import WarpFitter
from .utilities import warp_logger, addCommonWarpingArgumentsToParser, checkDirAndFixedArgs, getOctetsFromArguments
from .config import CONST
from ...utilities.misc import split_csv_to_list, split_csv_to_list_of_ints, split_csv_to_dict_of_floats, split_csv_to_dict_of_bounds
from ...utilities.config import CONST as UNIV_CONST
from argparse import ArgumentParser
import multiprocessing as mp
import pathlib, gc, cProfile

#################### FILE-SCOPE CONSTANTS ####################

#placeholder default arguments for command line options
DEFAULT_OVERLAPS    = '-999' 
DEFAULT_OCTETS      = '-999' 

#################### HELPER FUNCTIONS ####################

#helper function to check all command-line arguments
def checkArgs(args) :
    #make sure directories exist and fixed parameter choice is valid
    checkDirAndFixedArgs(args,parse=True)
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
        if not pathlib.Path.is_dir(pathlib.Path(args.threshold_file_dir)) :
            raise ValueError(f'ERROR: threshold_file_dir ({args.threshold_file_dir}) does not exist!')
        tfp = pathlib.Path(f'{args.threshold_file_dir}/{args.slideID}_{UNIV_CONST.BACKGROUND_THRESHOLD_TEXT_FILE_NAME_STEM}')
        if not pathlib.Path.is_file(tfp) :
            raise ValueError(f'ERROR: threshold_file_dir does not contain a threshold file for this slide ({tfp})!')
    #The user must specify either an octet run dir or a threshold file dir if they're not giving overlaps
    if args.overlaps==split_csv_to_list_of_ints(DEFAULT_OVERLAPS) and args.threshold_file_dir is None and args.octet_run_dir is None :
        raise ValueError('ERROR: must specify either an octet_run_dir or a threshold_file_dir!')
    #if the thresholding file dir and the octet dir are both provided the user needs to disambiguate
    if args.threshold_file_dir is not None and args.octet_run_dir is not None :
        raise ValueError('ERROR: cannot specify both an octet_run_dir and a threshold_file_dir!')

# Helper function to determine the list of overlaps
def getOverlaps(args) :
    #the threshold file must exist if it's to be used
    if args.threshold_file_dir is not None :
        if not pathlib.Path.is_dir(pathlib.Path(args.threshold_file_dir)) :
            raise ValueError(f'ERROR: threshold_file_dir ({args.threshold_file_dir}) does not exist!')
        tfp = pathlib.Path(f'{args.threshold_file_dir}/{args.slideID}_{UNIV_CONST.BACKGROUND_THRESHOLD_TEXT_FILE_NAME_STEM}')
        if not pathlib.Path.is_file(tfp) :
            raise ValueError(f'ERROR: threshold_file_dir does not contain a threshold file for this slide ({tfp})!')
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
        valid_octets = getOctetsFromArguments(args)
        if args.mode in ('fit', 'check_run', 'cProfile') and args.octets!=split_csv_to_list_of_ints(DEFAULT_OCTETS):
            for octet in valid_octets :
                if octet.p1_rect_n in args.octets or args.octets==[-1]:
                    warp_logger.info(f'Adding overlaps in octet surrounding rectangle {octet.p1_rect_n}...')
                    overlaps+=octet.overlap_ns
            if (args.octets!=[-1] and len(overlaps)!=8*len(args.octets)) or (args.octets==[-1] and len(overlaps)!=8*len(valid_octets)) :
                msg =f'specified octets ({args.octets}) did not result in the desired set of overlaps! '
                msg+=f'(asked for {len(args.octets)} octets but found {len(overlaps)} corresponding overlaps.)'
                msg+=f' There are {len(valid_octets)} octets to choose from.'
                raise ValueError(msg)
    return overlaps

#################### MAIN SCRIPT ####################

def main(args=None) :
    mp.freeze_support()
    if args is None :
        #define and get the command-line arguments
        parser = ArgumentParser()
        #positional arguments
        parser.add_argument('mode', help='Operation to perform', choices=['fit','find_octets','check_run','cProfile'])
        #add the common arguments
        addCommonWarpingArgumentsToParser(parser,job_organization=False)
        #additional group for how to figure out which overlaps will be used
        overlap_selection_group = parser.add_argument_group('overlap selection', 'what set of overlaps should be used?')
        overlap_selection_group.add_argument('--overlaps',       default=DEFAULT_OVERLAPS, type=split_csv_to_list_of_ints,         
                                             help='Comma-separated list of numbers (n) of the overlaps to use (two-element defines a range)')
        overlap_selection_group.add_argument('--octets',         default=DEFAULT_OCTETS,   type=split_csv_to_list_of_ints,         
                                             help='Comma-separated list of overlap octet indices (ordered by n of octet central rectangle) to use')
        #unique arguments
        parser.add_argument('--save_warp_fields', action='store_true',
                         help='Add this flag to save the warping fields for this fit and not just the result file')
        args = parser.parse_args(args=args)
    #apply some checks to the arguments to make sure they're valid
    checkArgs(args)
    #get the overlaps to use for fitting
    overlaps=getOverlaps(args)
    gc.collect()
    #setup and run
    if args.mode in ('fit', 'check_run', 'cProfile') :
        #how many overlaps will be used
        warp_logger.info(f'Will run fit using {len(overlaps)} total overlaps.')
        #make the WarpFitter Objects
        warp_logger.info('Initializing WarpFitter')
        fitter = WarpFitter(args.slideID,args.rawfile_top_dir,args.root_dir,args.workingdir,overlaps,args.layer)
        #check the run if that's what's being asked
        if args.mode in ('check_run') :
            fitter.checkFit(fixed=args.fixed,normalize=args.normalize,init_pars=args.init_pars,init_bounds=args.init_bounds,
                            float_p1p2_in_polish_fit=args.float_p1p2_to_polish,
                            max_radial_warp=args.max_radial_warp,max_tangential_warp=args.max_tangential_warp,
                            p1p2_polish_lasso_lambda=args.p1p2_polish_lasso_lambda,polish=True,save_fields=args.save_warp_fields)
        #otherwise actually run it
        elif args.mode in ('fit', 'cProfile') :
            #load the raw files
            warp_logger.info('Loading raw files')
            fitter.loadRawFiles(args.flatfield_file,args.exposure_time_offset_file)
             #fit the model to the data
            warp_logger.info('Running doFit')
            if args.mode == 'fit' :
                fitter.doFit(fixed=args.fixed,normalize=args.normalize,init_pars=args.init_pars,init_bounds=args.init_bounds,
                             float_p1p2_in_polish_fit=args.float_p1p2_to_polish,
                             max_radial_warp=args.max_radial_warp,max_tangential_warp=args.max_tangential_warp,
                             p1p2_polish_lasso_lambda=args.p1p2_polish_lasso_lambda,polish=True,
                             print_every=args.print_every,maxiter=args.max_iter,save_fields=args.save_warp_fields)
            elif args.mode == 'cProfile' :
                cProfile.run("""fitter.doFit(fixed=args.fixed,normalize=args.normalize,init_pars=args.init_pars,init_bounds=args.init_bounds,
                                float_p1p2_in_polish_fit=args.float_p1p2_to_polish,
                                max_radial_warp=args.max_radial_warp,max_tangential_warp=args.max_tangential_warp,
                                p1p2_polish_lasso_lambda=args.p1p2_polish_lasso_lambda,polish=True,
                                print_every=args.print_every,maxiter=args.max_iter,save_fields=args.save_warp_fields)""")

if __name__=='__main__' :
    main()
