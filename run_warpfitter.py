#imports 
from .warpfitter import WarpFitter
from argparse import ArgumentParser
import os, logging

#################### PARSE ARGUMENTS ####################

#parser callback function to split a string of comma-separated values into a list
def split_csv_to_list(value) :
    return value.split(',')

#parser callback function to split a string of comma-separated values into a list of integers
def split_csv_to_list_of_ints(value) :
    try :
        return [int(v) for v in value.split(',')]
    except ValueError :
        raise ValueError(f'Option value {value} is expected to be a comma-separated list of integers!')

#define and get the command-line arguments
parser = ArgumentParser()
#positional arguments
parser.add_argument('sample',      help='Name of the data sample to use')
parser.add_argument('rawfile_dir', help='Path to the directory containing the "[sample_name]/*.raw" files')
parser.add_argument('root1_dir',   help='Path to the directory containing "[sample name]/dbload" subdirectories')
#optional arguments
parser.add_argument('--working_dir',         default='warpfit_test',
    help='Path to (or name of) the working directory that will be created')
parser.add_argument('--overlaps',            default='-1',        type=split_csv_to_list_of_ints,         
    help='Comma-separated list of numbers (n) of the overlaps to use (two-element defines a range)')
parser.add_argument('--layers',              default='1',         type=split_csv_to_list_of_ints,         
    help='Comma-separated list of image layers to use')
parser.add_argument('--fixed',               default='',          type=split_csv_to_list,         
    help='Comma-separated list of parameters to keep fixed during fitting')
parser.add_argument('--max_radial_warp',     default=25.,         type=float,
    help='Maximum amount of radial warp to use for constraint')
parser.add_argument('--max_tangential_warp', default=25.,         type=float,
    help='Maximum amount of radial warp to use for constraint')
parser.add_argument('--print_every',         default=10,          type=int,
    help='Maximum amount of radial warp to use for constraint')
args = parser.parse_args()

#apply some checks to the arguments to make sure they're valid
#rawfile_dir/[sample] must exist
rawfile_dir = args.rawfile_dir if args.rawfile_dir.endswith(args.sample) else os.path.join(args.rawfile_dir,args.sample)
if not os.path.isdir(rawfile_dir) :
    raise ValueError(f'rawfile_dir argument ({rawfile_dir}) does not point to a valid directory!')
#root1 dir must exist
if not os.path.isdir(args.root1_dir) :
    raise ValueError(f'root1_dir argument ({args.root1_dir}) does not point to a valid directory!')
#root1 dir must be usable to find a metafile directory
metafile_dir = os.path.join(args.root1_dir,args.sample,'dbload')
if not os.path.isdir(metafile_dir) :
    raise ValueError(f'root1_dir ({args.root1_dir}) does not contain "[sample name]/dbload" subdirectories!')
#overlaps must be -1, a tuple, or a list
overlaps = args.overlaps
if len(overlaps)==1 :
    overlaps = overlaps[0]
    if overlaps!=-1 :
        raise ValueError(f'single overlaps argument ({args.overlaps}) must be -1 (to use all overlaps)!')
elif len(overlaps)==2 :
    overlaps=tuple(overlaps)
#the parameter fixing string must correspond to some combination of options
fix_cxcy = 'cx' in args.fixed and 'cy' in args.fixed
fix_fxfy = 'fx' in args.fixed and 'fy' in args.fixed
fix_k1k2 = 'k1' in args.fixed and 'k2' in args.fixed
fix_p1p2 = 'p1' in args.fixed and 'p2' in args.fixed
if args.fixed!=[''] and len(args.fixed)!=2*sum([fix_cxcy,fix_fxfy,fix_k1k2,fix_p1p2]) :
    raise ValueError(f'Fixed parameters argument ({args.fixed}) does not result in a valid fixed parameter condition!')

#################### RUN THE WARPFITTER ####################

logger = logging.getLogger("warpfitter")
#make the WarpFitter Objects
logger.info('Initializing WarpFitter')
fitter = WarpFitter(args.sample,rawfile_dir,metafile_dir,args.working_dir,overlaps,args.layers)
#load the raw files
logger.info('Loading raw files')
fitter.loadRawFiles()
#fit the model to the data
logger.info('Running doFit')
result = fitter.doFit(fix_cxcy=fix_cxcy,fix_fxfy=fix_fxfy,fix_k1k2=fix_k1k2,fix_p1p2=fix_p1p2,
                      max_radial_warp=args.max_radial_warp,max_tangential_warp=args.max_tangential_warp,
                      print_every=args.print_every)
logger.info('All done')

