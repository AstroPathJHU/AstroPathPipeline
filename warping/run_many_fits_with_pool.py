#imports
from .warp import CameraWarp
from .plotting import principalPointPlot, radWarpAmtPlots, radWarpParPlots, radWarpPCAPlots, warpFieldVariationPlots
from .utilities import warp_logger, addCommonWarpingArgumentsToParser, checkDirAndFixedArgs, getOctetsFromArguments, WarpFitResult, FieldLog
from .config import CONST
from ..utilities.tableio import readtable, writetable
from ..utilities.misc import cd
from argparse import ArgumentParser
import os, random, math, subprocess, multiprocessing as mp

#################### FILE-SCOPE CONSTANTS ####################

RUN_WARPFITTER_PREFIX = 'python -m microscopealignment.warping.run_warp_fitter' #part of the command referencing how to run run_warpfitter.py
JOB_DIR_STEM = 'warpfitter_batch'
POSITIONAL_PASSTHROUGH_ARG_NAMES = ['sample','rawfile_top_dir','metadata_top_dir']
PASSTHROUGH_ARG_NAMES = ['exposure_time_offset_file','flatfield_file','max_iter','fixed','normalize','init_pars','init_bounds','max_radial_warp']
PASSTHROUGH_ARG_NAMES+= ['max_tangential_warp','p1p2_polish_lasso_lambda','print_every','layer']
PASSTHROUGH_FLAG_NAMES = ['skip_exposure_time_correction','skip_flatfielding','float_p1p2_to_polish']

#################### HELPER FUNCTIONS ####################

#function that gets passed to the multiprocessing pool just runs run_warpfitter.py
def worker(cmd) :
    subprocess.call(cmd)

#helper function to make the list of job commands
def getListOfJobCommands(args) :
    #first split the octet selection argument to get how they should be chosen and 
    octet_select_method = (args.octet_selection.split('_')[0]).lower()
    if octet_select_method not in ['first','random'] :
        raise ValueError(f'ERROR: octet_selection argument ({args.octet_selection}) is not valid! Use "first_n" or "random_n".')
    try :
        n_octets_per_job = int(args.octet_selection.split('_')[1])
    except ValueError :
        raise ValueError(f'ERROR: octet_selection argument ({args.octet_selection}) is not valid! Use "first_n" or "random_n".')
    #find the valid octets in the samples and order them by the # of their center rectangle
    all_octets = getOctetsFromArguments(args)
    #make sure that the number of octets per job and the number of jobs will work for this sample
    if args.njobs*n_octets_per_job<1 or args.njobs*n_octets_per_job>len(all_octets) :
        raise ValueError(f"""ERROR: Sample {args.sample} has {len(all_octets)} total valid octets, but you asked for {args.njobs} jobs 
                             with {n_octets_per_job} octets per job!""")
    #build the list of commands
    job_cmds = []; workingdir_names = []
    argvars = vars(args)
    cmd_base=f'{RUN_WARPFITTER_PREFIX} fit '
    for ppan in POSITIONAL_PASSTHROUGH_ARG_NAMES :
        cmd_base+=f'{argvars[ppan]} '
    for i in range(args.njobs) :
        thisjobdirname = JOB_DIR_STEM+'_octets'
        thisjoboctetstring = ''
        for j in range(n_octets_per_job) :
            index = 0 if octet_select_method=='first' else random.randint(0,len(all_octets)-1)
            this_octet_number = (all_octets.pop(index)).p1_rect_n
            thisjobdirname+=f'_{this_octet_number}'
            thisjoboctetstring+=f'{this_octet_number},'
        thisjobworkingdir = os.path.join(args.workingdir,thisjobdirname)
        workingdir_names.append(thisjobworkingdir)
        octet_run_dir = args.octet_run_dir if args.octet_run_dir is not None else thisjobworkingdir
        thisjobcmdstring = f'{cmd_base} {thisjobworkingdir} --octet_run_dir {octet_run_dir} --octets {thisjoboctetstring[:-1]} '
        for pan in PASSTHROUGH_ARG_NAMES :
            if argvars[pan] is not None :
                thisjobcmdstring+=f'--{pan} {argvars[pan]} '
        for pfn in PASSTHROUGH_FLAG_NAMES :
            if argvars[pfn] :
                thisjobcmdstring+=f'--{pfn} '
        thisjobcmdstring+='--n_threads 1'
        job_cmds.append(thisjobcmdstring)
    return job_cmds, workingdir_names

#################### MAIN SCRIPT ####################

if __name__=='__main__' :
    mp.freeze_support()
    #define and get the command-line arguments
    parser = ArgumentParser()
    #add the positional mode argument
    parser.add_argument('mode', help='What to do', choices=['fit','check_run'])
    #add the common arguments
    addCommonWarpingArgumentsToParser(parser)
    #additional positional arguments
    parser.add_argument('njobs', help='Number of jobs to run', type=int)
    args = parser.parse_args()
    #check some of the basic arguments
    checkDirAndFixedArgs(args)
    #get the list of all the job commands
    job_cmds, dirnames = getListOfJobCommands(args)
    #run the first command in check_run mode to make sure that things will work when they do get going
    warp_logger.info('TESTING first command in the list...')
    test_run_command = f'{RUN_WARPFITTER_PREFIX} check_run {(job_cmds[0])[(len(RUN_WARPFITTER_PREFIX)+len(" fit ")):]}'
    print(test_run_command)
    subprocess.call(test_run_command)
    warp_logger.info('TESTING done')
    if args.mode=='fit' :
        #run all the job commands on a pool of workers
        nworkers = min(mp.cpu_count(),args.njobs) if args.workers is None else min(args.workers,args.njobs,mp.cpu_count())
        warp_logger.info(f'WILL RUN {args.njobs} COMMANDS ON A POOL OF {nworkers} WORKERS:')
        pool = mp.Pool(nworkers)
        for i,cmd in enumerate(job_cmds,start=1) :
            warp_logger.info(f'  command {i}: {cmd}')
            pool.apply_async(worker,args=(cmd,))
        pool.close()
        warp_logger.info('POOL CLOSED; BATCH RUNNING!!')
        pool.join()
        warp_logger.info('All jobs in the pool have finished! : ) Collecting results....')
        #write out a list of all the individual results
        results = []
        for dirname in dirnames :
            results.append((readtable(os.path.join(dirname,CONST.FIT_RESULT_CSV_FILE_NAME),WarpFitResult))[0])
        with cd(args.workingdir) :
            writetable(f'all_results_{os.path.basename(os.path.normpath(args.workingdir))}.csv',results)
        if os.path.isfile(os.path.join(args.workingdir,f'all_results_{os.path.basename(os.path.normpath(args.workingdir))}.csv')) :
            for dirname in dirnames :
                os.remove(os.path.join(dirname,CONST.FIT_RESULT_CSV_FILE_NAME))
        #write out some plots
        plot_name_stem = f'{os.path.basename(os.path.normpath(args.workingdir))}'
        with cd(args.workingdir) :
            plotdirname = 'batch_plots'
            if not os.path.isdir(plotdirname) :
                os.mkdir(plotdirname)
            with cd(plotdirname) :
                principalPointPlot(results,save_stem=plot_name_stem)
                radWarpAmtPlots(results,save_stem=plot_name_stem)
                radWarpParPlots(results,save_stem=plot_name_stem)
                if not ('k1' in args.fixed and 'k2' in args.fixed and 'k3' in args.fixed) :
                    radWarpPCAPlots(results,weighted=False,save_stem=plot_name_stem)
                    radWarpPCAPlots(results,weighted=True,save_stem=plot_name_stem)
                warpFieldVariationPlots(results,save_stem=plot_name_stem)
        #get the weighted average parameters over all the results that reduced the cost
        warp_logger.info('Writing out info for weighted average warp....')
        w_cx = 0.; w_cy = 0.; w_fx = 0.; w_fy = 0.
        w_k1 = 0.; w_k2 = 0.; w_k3 = 0.; w_p1 = 0.; w_p2 = 0.
        sum_weights = 0.
        for result in [r for r in results if r.cost_reduction>0] :
            w = result.cost_reduction
            w_cx+=(w*result.cx); w_cy+=(w*result.cy); w_fx+=(w*result.fx); w_fy+=(w*result.fy)
            w_k1+=(w*result.k1); w_k2+=(w*result.k2); w_k3+=(w*result.k3)
            w_p1+=(w*result.p1); w_p2+=(w*result.p2)
            sum_weights+=w
        if sum_weights!=0. :
            w_cx/=sum_weights; w_cy/=sum_weights; w_fx/=sum_weights; w_fy/=sum_weights
            w_k1/=sum_weights; w_k2/=sum_weights; w_k3/=sum_weights; w_p1/=sum_weights; w_p2/=sum_weights
        #make a warp from these w average parameters and write out its info
        w_avg_warp = CameraWarp(results[0].n,results[0].m,w_cx,w_cy,w_fx,w_fy,w_k1,w_k2,w_k3,w_p1,w_p2)
        w_avg_result = WarpFitResult()
        w_avg_result.dirname = args.workingdir
        w_avg_result.n  = results[0].n
        w_avg_result.m  = results[0].m
        w_avg_result.cx = w_avg_warp.cx
        w_avg_result.cy = w_avg_warp.cy
        w_avg_result.fx = w_avg_warp.fx
        w_avg_result.fy = w_avg_warp.fy
        w_avg_result.k1 = w_avg_warp.k1
        w_avg_result.k2 = w_avg_warp.k2
        w_avg_result.k3 = w_avg_warp.k3
        w_avg_result.p1 = w_avg_warp.p1
        w_avg_result.p2 = w_avg_warp.p2
        max_r_x, max_r_y = w_avg_warp._getMaxDistanceCoords(w_avg_warp.cx,w_avg_warp.cy)
        w_avg_result.max_r_x_coord  = max_r_x
        w_avg_result.max_r_y_coord  = max_r_y
        w_avg_result.max_r          = math.sqrt((max_r_x)**2+(max_r_y)**2)
        w_avg_result.max_rad_warp   = w_avg_warp.maxRadialDistortAmount(None)
        w_avg_result.max_tan_warp   = w_avg_warp.maxTangentialDistortAmount(None)
        with cd(args.workingdir) :
            writetable(f'{os.path.basename(os.path.normpath(args.workingdir))}_weighted_average_{CONST.FIT_RESULT_CSV_FILE_NAME}',[w_avg_result])
            w_avg_warp.writeOutWarpFields(os.path.basename(os.path.normpath(args.workingdir)))
        #aggregate the different metadata summary and field log files into one
        all_field_logs = []
        for dirname in dirnames :
            all_field_logs+=((readtable(os.path.join(dirname,f'field_log_{os.path.basename(os.path.normpath(dirname))}.csv'),FieldLog)))
        with cd(args.workingdir) :
            writetable(f'field_log_{os.path.basename(os.path.normpath(args.workingdir))}.csv',all_field_logs)
    warp_logger.info('Done.')

