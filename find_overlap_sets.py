#imports 
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
parser.add_argument('--layers',              default='1',         type=split_csv_to_list_of_ints,         
    help='Comma-separated list of image layers to use')
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

#################### MAIN SCRIPT ####################

logger = logging.getLogger("find_overlap_sets")


