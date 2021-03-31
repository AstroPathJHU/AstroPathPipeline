#imports
from .utilities import warp_logger, WarpingError, WarpFitResult, WarpingSummary, addCommonWarpingArgumentsToParser, checkDirArgs, getOctetsFromArguments
from .config import CONST
from ...utilities.tableio import readtable, writetable
from ...utilities.misc import cd
from ...utilities.config import CONST as UNIV_CONST
import numpy as np
from argparse import ArgumentParser
import os, subprocess, random, shutil

#################### FILE-SCOPE CONSTANTS ####################
INITIAL_PATTERN_DIR_STEM = 'warping_initial_pattern'
PRINCIPAL_POINT_DIR_STEM = 'warping_center_principal_point'
FINAL_PATTERN_DIR_STEM   = 'warping_final_pattern'
RUN_MANY_FITS_CMD_BASE = 'run_many_warp_fits_with_pool'
POSITIONAL_PASSTHROUGH_ARG_NAMES = ['mode','slideID','rawfile_top_dir','root_dir']
PASSTHROUGH_ARG_NAMES = ['exposure_time_offset_file','flatfield_file','layer','workers']
PASSTHROUGH_FLAG_NAMES = ['skip_exposure_time_correction','skip_flatfielding']

#################### HELPER FUNCTIONS ####################

#helper function to make sure the command line arguments are alright
def checkArgs(args) :
    #check to make sure the directories exist and the 'fixed' argument is okay
    checkDirArgs(args)
    #tell the user what's going to happen based on the mode/octet splitting arguments
    if args.mode=='fit' :
        warp_logger.info(f'Three groups of fits will be performed, using {args.workers} CPUs (max) each, to find the warping pattern for {args.slideID}:')
        warp_logger.info(f'{args.initial_pattern_octets} octets will be used to find the initial general pattern')
        warp_logger.info(f'{args.principal_point_octets} octets will be used to define the center principal point location')
        warp_logger.info(f'{args.final_pattern_octets} octets will be used to find the overall best pattern')
    elif args.mode=='find_octets' :
        warp_logger.info(f'The set of valid octets to use will be found for {args.slideID}')
    elif args.mode=='check_run' :
        warp_logger.info(f'Octets for {args.slideID} will be found/split, and a test command will be run for the first group of fits.')

#helper function to set up the three fit group working directories
def setUpFitDirectories(args) :
    #find the octets for the slide
    octets = getOctetsFromArguments(args)
    #randomize and split the octets into groups for the three fits
    if len(octets) < args.initial_pattern_octets+args.principal_point_octets+args.final_pattern_octets :
        msg = f'ERROR: There are {len(octets)} valid octets for {args.slideID} but you requested using {args.initial_pattern_octets}, then {args.principal_point_octets},'
        msg+= f' and then {args.final_pattern_octets} in the fit groups, respectively!'
        raise WarpingError(msg)
    random.shuffle(octets)
    octets_1 = octets[:args.initial_pattern_octets];            start = args.initial_pattern_octets
    octets_2 = octets[start:start+args.principal_point_octets]; start+= args.principal_point_octets
    octets_3 = octets[start:start+args.final_pattern_octets]
    #create the working directories for the three fit groups and copy the subsets of the octets files
    wdn_1 = f'{INITIAL_PATTERN_DIR_STEM}_{args.slideID}_{len(octets_1)}_octets'
    wdn_2 = f'{PRINCIPAL_POINT_DIR_STEM}_{args.slideID}_{len(octets_2)}_octets'
    wdn_3 = f'{FINAL_PATTERN_DIR_STEM}_{args.slideID}_{len(octets_3)}_octets'
    wdns = [wdn_1,wdn_2,wdn_3]
    ols = [octets_1,octets_2,octets_3]
    with cd(args.workingdir) :
        for wdn,o in zip(wdns,ols) :
            os.makedirs(wdn,exist_ok=True)
            with cd(wdn) :
                writetable(f'{args.slideID}{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM}',o)
    #return the names of the created directories
    return tuple(wdns)

#helper function to make the command to run for the initial pattern fit group
def getInitialPatternFitCmd(wdn,args) :
    #start with the call to run_many_fits_with_pool
    cmd = RUN_MANY_FITS_CMD_BASE
    #add the positional arguments
    argvars = vars(args)
    for ppan in POSITIONAL_PASSTHROUGH_ARG_NAMES :
        cmd+=f' {argvars[ppan]} '
    #add the working directory argument
    this_job_dir_path = os.path.abspath(os.path.join(args.workingdir,wdn))
    cmd+=f'{this_job_dir_path} '
    #add the number of jobs positional argument
    cmd+=f'{args.initial_pattern_octets} '
    #add the passthrough arguments and flags
    for pan in PASSTHROUGH_ARG_NAMES :
        if argvars[pan] is not None :
            cmd+=f'--{pan} {argvars[pan]} '
    for pfn in PASSTHROUGH_FLAG_NAMES :
        if argvars[pfn] :
            cmd+=f'--{pfn} '
    #add the number of iterations to run
    cmd+=f'--max_iter {args.initial_pattern_max_iters} '
    #the octets are in the working directory
    cmd+=f'--octet_run_dir {this_job_dir_path} '
    #select the first single octet for every job since we've already split up the octets for the slide
    cmd+='--octet_selection first_1 '
    #fix the focal lengths and tangential warping parameters
    cmd+='--fixed fx,fy,p1,p2'
    #return the command
    return cmd

#helper function to make the command to run for the principal point fit group
def getPrincipalPointFitCmd(wdn,args,k1,k2,k3) :
    #start with the call to run_many_fits_with_pool
    cmd = RUN_MANY_FITS_CMD_BASE
    #add the positional arguments
    argvars = vars(args)
    for ppan in POSITIONAL_PASSTHROUGH_ARG_NAMES :
        cmd+=f' {argvars[ppan]} '
    #add the working directory argument
    this_job_dir_path = os.path.abspath(os.path.join(args.workingdir,wdn))
    cmd+=f'{this_job_dir_path} '
    #add the number of jobs positional argument
    cmd+=f'{args.principal_point_octets} '
    #add the passthrough arguments and flags
    for pan in PASSTHROUGH_ARG_NAMES :
        if argvars[pan] is not None :
            cmd+=f'--{pan} {argvars[pan]} '
    for pfn in PASSTHROUGH_FLAG_NAMES :
        if argvars[pfn] :
            cmd+=f'--{pfn} '
    #add the number of iterations to run
    cmd+=f'--max_iter {args.principal_point_max_iters} '
    #the octets are in the working directory
    cmd+=f'--octet_run_dir {this_job_dir_path} '
    #select the first single octet for every job since we've already split up the octets for the slide
    cmd+='--octet_selection first_1 '
    #fix the focal lengths and warping parameters
    cmd+='--fixed fx,fy,k1,k2,k3,p1,p2 '
    #add the initial radial warping parameters from the weighted avg. of the last group of fits
    cmd+=f'--init_pars k1={k1},k2={k2},k3={k3}'
    #return the command
    return cmd

#helper function to make the command to run for the final pattern fit group
def getFinalPatternFitCmd(wdn,args,k1,k2,k3,cx,cx_err,cy,cy_err) :
    #start with the call to run_many_fits_with_pool
    cmd = RUN_MANY_FITS_CMD_BASE
    #add the positional arguments
    argvars = vars(args)
    for ppan in POSITIONAL_PASSTHROUGH_ARG_NAMES :
        cmd+=f' {argvars[ppan]} '
    #add the working directory argument
    this_job_dir_path = os.path.abspath(os.path.join(args.workingdir,wdn))
    cmd+=f'{this_job_dir_path} '
    #add the number of jobs positional argument
    cmd+=f'{args.final_pattern_octets} '
    #add the passthrough arguments and flags
    for pan in PASSTHROUGH_ARG_NAMES :
        if argvars[pan] is not None :
            cmd+=f'--{pan} {argvars[pan]} '
    for pfn in PASSTHROUGH_FLAG_NAMES :
        if argvars[pfn] :
            cmd+=f'--{pfn} '
    #add the number of iterations to run
    cmd+=f'--max_iter {args.final_pattern_max_iters} '
    #the octets are in the working directory
    cmd+=f'--octet_run_dir {this_job_dir_path} '
    #select the first single octet for every job since we've already split up the octets for the slide
    cmd+='--octet_selection first_1 '
    #fix the focal lengths and tangential warping parameters
    cmd+='--fixed fx,fy,p1,p2 '
    #add the initial radial warping parameters from the weighted avg. of the last group of fits
    cmd+=f'--init_pars cx={cx},cy={cy},k1={k1},k2={k2},k3={k3} '
    #add the bounds on the center point
    cmd+=f'--init_bounds cx={cx-cx_err}:{cx+cx_err},cy={cy-cy_err}:{cy+cy_err}'
    #return the command
    return cmd

#################### MAIN SCRIPT ####################

def main(args=None) :
    #define and get the command-line arguments
    parser = ArgumentParser()
    #add the positional mode argument
    parser.add_argument('mode', help='What to do', choices=['fit','find_octets','check_run'])
    #add only a few of the common arguments
    addCommonWarpingArgumentsToParser(parser,fit=False,fitpars=False,job_organization=False)
    #add the positional number of workers argument
    parser.add_argument('workers', default=None, type=int, help='Max # of CPUs to use in the multiprocessing pools (defaults to all available)')
    #arguments for how to split the total group of octets
    octet_splitting_group = parser.add_argument_group('octet splitting', 'how to split the total set of octets for each of the three fit groups')
    octet_splitting_group.add_argument('--initial_pattern_octets', type=int, default=50,
                                       help='Number of octets to use in the initial pattern fits')
    octet_splitting_group.add_argument('--principal_point_octets', type=int, default=50,
                                       help='Number of octets to use in the principal point location fits')
    octet_splitting_group.add_argument('--final_pattern_octets',   type=int, default=100,
                                       help='Number of octets to use in the final pattern fits')
    #arguments for how many iterations to run at maximum in the groups of fits
    max_iters_group = parser.add_argument_group('max iterations', 'how many iterations to run at max for minimization in each of the three fit groups')
    max_iters_group.add_argument('--initial_pattern_max_iters', type=int, default=3000,
                                       help='Max # of iterations to run in the initial pattern fits')
    max_iters_group.add_argument('--principal_point_max_iters', type=int, default=8000,
                                       help='Max # of iterations to run in the principal point location fits')
    max_iters_group.add_argument('--final_pattern_max_iters',   type=int, default=10000,
                                       help='Max # of iterations to run in the final pattern fits')
    args = parser.parse_args(args=args)
    #make sure the arguments are alright
    checkArgs(args)
    #set up the three directories with their octet groupings
    dirname_1, dirname_2, dirname_3 = setUpFitDirectories(args)
    if args.mode!='find_octets' :
        #start up the file that will have the commands written into it
        cmd_file_path = os.path.abspath(os.path.join(args.workingdir,'fit_group_commands.txt'))
        #get the command for the initial pattern fits and run it
        cmd_1 = getInitialPatternFitCmd(dirname_1,args)
        with open(cmd_file_path,'w') as fp :
            fp.write(f'{cmd_1}\n\n')
        subprocess.call(cmd_1)
    if args.mode=='fit' :
        #figure out the radial warping parameters to use when finding the principal point
        w_avg_res_1 = (readtable(os.path.abspath(os.path.join(args.workingdir,dirname_1,
                                                            f'{dirname_1}_weighted_average_{CONST.WARPING_SUMMARY_CSV_FILE_NAME}')),WarpingSummary))[0]
        init_k1 = w_avg_res_1.k1; init_k2 = w_avg_res_1.k2; init_k3 = w_avg_res_1.k3
        #get the command for the central principal point fits and run it
        cmd_2 = getPrincipalPointFitCmd(dirname_2,args,init_k1,init_k2,init_k3)
        with open(cmd_file_path,'a') as fp :
            fp.write(f'{cmd_2}\n\n')
        subprocess.call(cmd_2)
        #find the weighted mean of the principal point and its error from the previous run results 
        all_results_2 = readtable(os.path.abspath(os.path.join(args.workingdir,dirname_2,f'all_results_{dirname_2}.csv')),WarpFitResult)
        w_cx = 0.; w_cy = 0.; sw = 0.; sw2 = 0.
        for r in all_results_2 :
            w = r.cost_reduction
            if w<0 :
                continue
            w_cx+=(w*r.cx); w_cy+=(w*r.cy); sw+=w; sw2+=w**2
        w_cx/=sw; w_cy/=sw
        w_cx_e = np.sqrt(((np.std([r.cx for r in all_results_2 if r.cost_reduction>0])**2)*sw2)/(sw**2))
        w_cy_e = np.sqrt(((np.std([r.cx for r in all_results_2 if r.cost_reduction>0])**2)*sw2)/(sw**2))
        #get the command for the final pattern fits and run it
        cmd_3 = getFinalPatternFitCmd(dirname_3,args,init_k1,init_k2,init_k3,w_cx,w_cx_e,w_cy,w_cy_e)
        with open(cmd_file_path,'a') as fp :
            fp.write(f'{cmd_3}\n')
        subprocess.call(cmd_3)
        warp_logger.info(f'All fits for {args.slideID} warping pattern completed')
        #move and rename the final warping field and weighted average result files from the last fit
        old_w_avg_fit_result_fp = os.path.abspath(os.path.join(args.workingdir,dirname_3,f'{dirname_3}_weighted_average_{CONST.WARPING_SUMMARY_CSV_FILE_NAME}'))
        new_w_avg_fit_result_fp = os.path.abspath(os.path.join(args.workingdir,
                                                f'{os.path.basename(os.path.normpath(args.workingdir))}_weighted_average_{CONST.WARPING_SUMMARY_CSV_FILE_NAME}'))
        shutil.move(old_w_avg_fit_result_fp,new_w_avg_fit_result_fp)
        old_w_avg_dx_wf_fp = os.path.abspath(os.path.join(args.workingdir,dirname_3,f'{UNIV_CONST.X_WARP_BIN_FILENAME}_{dirname_3}.bin'))
        old_w_avg_dy_wf_fp = os.path.abspath(os.path.join(args.workingdir,dirname_3,f'{UNIV_CONST.Y_WARP_BIN_FILENAME}_{dirname_3}.bin'))
        new_w_avg_dx_wf_fp = os.path.abspath(os.path.join(args.workingdir,
                                                f'{UNIV_CONST.X_WARP_BIN_FILENAME}_{os.path.basename(os.path.normpath(args.workingdir))}.bin'))
        new_w_avg_dy_wf_fp = os.path.abspath(os.path.join(args.workingdir,
                                                f'{UNIV_CONST.Y_WARP_BIN_FILENAME}_{os.path.basename(os.path.normpath(args.workingdir))}.bin'))
        shutil.move(old_w_avg_dx_wf_fp,new_w_avg_dx_wf_fp)
        shutil.move(old_w_avg_dy_wf_fp,new_w_avg_dy_wf_fp)
    warp_logger.info('Done')

if __name__=='__main__' :
    main()

