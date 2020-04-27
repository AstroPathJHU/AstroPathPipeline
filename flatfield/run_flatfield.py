#imports
from argparse import ArgumentParser
from ..utilities.misc import cd, split_csv_to_list_of_ints

#################### HELPER FUNCTIONS ####################

#helper function to make sure arguments are valid
def checkArgs(a) :
	#make sure raw image dimensions is the right size
	try:
		assert len(a.raw_image_dims)==3
	except AssertionError :
		raise ValueError(f'ERROR: Raw image dimensions ({a.raw_image_dims}) must be a list of three integers in x,y,#layers!')

#################### MAIN SCRIPT ####################
def main() :
	#define and get the command-line arguments
    parser = ArgumentParser()
    #positional arguments
    parser.add_argument('samplename_csv_file', help='Path to the csv file that lists the sample names to include')
    parser.add_argument('rawfile_top_dir',     help='Path to directory that holds each of the [samplename] directories that contain raw files')
    #optional arguments
    parser.add_argument('--raw_image_dims', default=[1344,1004,35], type=split_csv_to_list_of_ints,
        help='Comma-separated list of raw image dimensions in (x, y, #layers)')
    parser.add_argument('--layers',         default=[-1],           type=split_csv_to_list_of_ints,         
        help='Comma-separated list of image layer numbers to consider (default -1 means all layers)')
    parser.add_argument('--max_images',     default=-1,             type=int,         
        help='Number of images to load from the inputted list of samples')
    args = parser.parse_args()
    checkArgs(args)

if __name__=='__main__' :
    main()
    