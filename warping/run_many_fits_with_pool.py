#imports
from argparse import ArgumentParser
from .run_warpfitter import getSampleOctets, checkDirAndFixedArgs
from ..utilities.misc import split_csv_to_list
import os, random, multiprocessing as mp

#function that gets passed to the multiprocessing pool just runs run_warpfitter.py
def worker(cmd) :
	os.system(cmd)

def main() :
	#define and get the command-line arguments
	parser = ArgumentParser()
	#positional arguments
	parser.add_argument('sample',      help='Name of the data sample to use')
	parser.add_argument('rawfile_dir', help='Path to the directory containing the "[sample_name]/*.raw" files')
	parser.add_argument('root1_dir',   help='Path to the directory containing "[sample name]/dbload" subdirectories')
	parser.add_argument('root2_dir',   help='Path to the directory containing the "[sample_name]/*.fw01" files to use for initial alignment')
	parser.add_argument('njobs',       help='Number of jobs to run', type=int)
	#optional arguments
	parser.add_argument('--working_dir',         default='warpfit_pool',
	    help='Name of the working directory that will be created to hold output of all jobs')
	parser.add_argument('--job_dir_stem',         default='warpfit_test',
	    help="Stem of each job's working directory (_octets_*) will be appended")
	parser.add_argument('--octet_selection',       default='random_2',
		help='String for how to select octets for each job: "first_n" or "random_n".')
	parser.add_argument('--workers',            default=None,        type=int,
	    help='Number of CPUs to use in the multiprocessing pool (defaults to all available)')
	parser.add_argument('--layer',               default='1',         type=int,         
	    help='Image layer to use (indexed from 1)')
	parser.add_argument('--fixed',               default=['fx','fy'],          type=split_csv_to_list,         
	    help='Comma-separated list of parameters to keep fixed during fitting')
	parser.add_argument('--float_p1p2_to_polish', action='store_true',
        help='Add this flag to float p1 and p2 in the polishing minimization regardless of whether they are in the list of fixed parameters')
	parser.add_argument('--max_radial_warp',     default=8.,         type=float,
	    help='Maximum amount of radial warp to use for constraint')
	parser.add_argument('--max_tangential_warp', default=4.,         type=float,
	    help='Maximum amount of radial warp to use for constraint')
	parser.add_argument('--lasso_lambda',         default=0.0,         type=float,
        help='Lambda magnitude parameter for the LASSO constraint on p1 and p2 if those parameters are to float in the polishing minimization')
	parser.add_argument('--print_every',         default=1000,          type=int,
	    help='How many iterations to wait between printing minimization progress')
	parser.add_argument('--max_iter',            default=1000,        type=int,
	    help='Maximum number of iterations for differential_evolution and for minimize.trust-constr')
	args = parser.parse_args()

	#make sure the directory and fixed parameter arguments are valid
	checkDirAndFixedArgs(args)
	fixedstring='--fixed '
	for fixedpar in args.fixed :
		fixedstring+=f'{fixedpar},'

	#split the octet selection argument and make sure it's valid
	octet_select_method = (args.octet_selection.split('_')[0]).lower()
	try :
		n_octets_per_job = int(args.octet_selection.split('_')[1])
	except ValueError :
		raise ValueError(f'octet_selection argument ({args.octet_selection}) is not valid! Use "first_n" or "random_n".')
	if octet_select_method not in ['first','random'] :
		raise ValueError(f'octet_selection argument ({args.octet_selection}) is not valid! Use "first_n" or "random_n".')

	#make the batch directory
	if not os.path.isdir(os.path.join(os.getcwd(),args.working_dir)) :
		os.mkdir(args.working_dir)

	#start by ordering the list of octets for this sample by the # of their center rectangle
	all_octets_dict, _ = getSampleOctets(args.root1_dir,args.root2_dir,args.sample,args.working_dir,save_plots=False)
	all_octets_numbers = list(range(1,len(all_octets_dict.items())+1))

	#make sure that the number of octets per job and the number of jobs will work for this sample
	if args.njobs*n_octets_per_job<1 or args.njobs*n_octets_per_job>len(all_octets_numbers) :
		raise ValueError(f'Sample {args.sample} has {len(all_octets_dict)} total octets, but you asked for {args.njobs} jobs with {n_octets_per_job} octets per job!')

	#build the list of commands to pass to os.system to run run_warpfitter.py
	cmd_base=f'python -m microscopealignment.warping.run_warpfitter fit {args.sample} {args.rawfile_dir} {args.root1_dir} {args.root2_dir}'
	cmd_base+=f' --layer {args.layer}'
	cmd_base+=f' {fixedstring[:-1]}'
	if args.float_p1p2_to_polish :
		cmd_base+=' --float_p1p2_to_polish'
	cmd_base+=f' --max_radial_warp {args.max_radial_warp}'
	cmd_base+=f' --max_tangential_warp {args.max_tangential_warp}'
	cmd_base+=f' --lasso_lambda {args.lasso_lambda}'
	cmd_base+=f' --print_every {args.print_every}'
	cmd_base+=f' --max_iter {args.max_iter}'
	job_cmds = []
	for i in range(args.njobs) :
		thisjobdirname = args.job_dir_stem+'_octets'
		thisjoboctetstring = ''
		for j in range(n_octets_per_job) :
			index = 0 if octet_select_method=='first' else random.randint(0,len(all_octets_numbers)-1)
			this_octet_number = all_octets_numbers.pop(index)
			thisjobdirname+=f'_{this_octet_number}'
			thisjoboctetstring+=f'{this_octet_number},'
		thisjobworkingdir = os.path.join(args.working_dir,thisjobdirname)
		thisjobcmdstring = f'{cmd_base} --working_dir {thisjobworkingdir} --octets {thisjoboctetstring[:-1]}'
		job_cmds.append(thisjobcmdstring)

	#run all the job commands on a pool of workers
	nworkers = min(mp.cpu_count(),args.njobs) if args.workers is None else min(args.workers,args.njobs)
	print(f'WILL RUN {args.njobs} COMMANDS ON A POOL OF {nworkers} WORKERS:')
	pool = mp.Pool(nworkers)
	for i,cmd in enumerate(job_cmds,start=1) :
		print(f'  command {i}: {cmd}')
		pool.apply_async(worker,args=(cmd,))
	pool.close()
	print('POOL CLOSED; BATCH RUNNING!!')
	pool.join()
	print('All jobs in the pool have finished! : )')

if __name__=='__main__' :
	mp.freeze_support()
	main()
