#imports
from .config import flatfield_logger
from ..utilities.img_file_io import getImageHWLFromXMLFile
from ..utilities.misc import cd, split_csv_to_list, split_csv_to_list_of_ints
from argparse import ArgumentParser
import os, csv, random

#################### HELPER FUNCTIONS ####################

#helper function to make sure arguments are valid
def checkArgs(a) :
    #make sure that the raw file directory exists
    if not os.path.isdir(a.rawfile_top_dir) :
        raise ValueError(f'ERROR: Raw file directory {a.rawfile_top_dir} does not exist!')
    #create the working directory if it doesn't already exist
    if not os.path.isdir(a.workingdir_name) :
        os.mkdir(a.workingdir_name)
    #make sure that the user didn't ask to save the masking plots while skipping the masking
    if a.skip_masking and a.save_masking_plots :
        raise ValueError('ERROR: cannot save masking plots if masking will be skipped')

#helper function to get the list of sample names to run on
def getAllSampleNames(a) :
    #get them from the .csv file if requested
    if '.csv' in a.samplenames :
        #make sure that the samplename CSV file exists
        if not os.path.isfile(a.samplenames) :
            raise ValueError(f'ERROR: sample name CSV file {a.samplenames} does not exist!')
        with open(a.samplenames,) as f:
            reader = csv.reader(f)
            all_sample_names = (list(reader))[0]
    #otherwise they should just be listed in the command line argument
    else :
        all_sample_names = split_csv_to_list(a.samplenames)
    return all_sample_names

#helper function to get the list of filepaths and associated sample names to run on based on the selection method and number of images requested
def getFilepathsAndSampleNamesToRun(a) :
    #get all the possible sample names
    all_sample_names = getAllSampleNames(a)
    #make sure the directories all exist
    for sn in all_sample_names :
        if not os.path.isdir(os.path.join(a.rawfile_top_dir,sn)) :
            raise ValueError(f'ERROR: sample directory {os.path.join(a.rawfile_top_dir,sn)} does not exist!')
    #get the (sorted) full list of file names in the sample to choose from
    all_image_filepaths = []
    #iterate over the samples
    for sn in all_sample_names :
        #get the list of all the filenames for this sample
        with cd(os.path.join(a.rawfile_top_dir,sn)) :
            this_sample_image_filepaths = [os.path.join(a.rawfile_top_dir,sn,fn) 
                                            for fn in glob.glob(f'{sn}_[[]*,*[]]{a.rawfile_ext}')]
            all_image_filepaths+=this_sample_image_filepaths
    all_image_filepaths.sort()
    #select the total filepaths/sample names to run on
    filepaths_to_run = None
    logstring='Will run on a sample of'
    if min(a.max_images,len(all_image_filepaths)) in [-1,len(all_image_filepaths)] :
        logstring+=f' all {len(all_image_filepaths)} images' 
        if min(a.max_images,len(all_image_filepaths)) == len(all_image_filepaths) :
            logstring+=f' (not enough images in the sample(s) to deliver all {a.max_images} requested)'
        flatfield_logger.info(logstring)
        return all_image_filepaths
    if a.selection=='first' :
        filepaths_to_run=all_image_filepaths[:a.max_images]
        logstring+=f' the first {a.max_images} images'
    elif a.selection=='last' :
        filepaths_to_run=all_image_filepaths[-(a.max_images):]
        logstring+=f' the last {a.max_images} images'
    elif a.selection=='random' :
        random.shuffle(all_image_filepaths)
        filepaths_to_run=all_image_filepaths[:a.max_images]
        logstring+=f' {a.max_images} randomly-chosen images'
    flatfield_logger.info(logstring)
    #figure out the samples from which those files are coming
    samplenames_to_run = []
    for fp in filepaths_to_run :
        this_fp_sn = ((fp.split(os.path.sep)[-1]).split('_')[0])
        if this_fp_sn not in samplenames_to_run :
            samplenames_to_run.append(this_fp_sn)
    logstring = f'Images are sourced from {len(samplenames_to_run)}'
    if len(samplenames_to_run)>1 :
        logstring+=' different samples:'
    else :
        logstring+=' sample'
    logstring+=': '
    for sn in samplenames_to_run :
        logstring+=f'{sn}, '
    flatfield_logger.info(logstring[:-2])
    #return the lists of filepaths and samplenames
    return filepaths_to_run, samplenames_to_run

#################### MAIN SCRIPT ####################
def main() :
    #define and get the command-line arguments
    parser = ArgumentParser()
    #positional arguments
    parser.add_argument('samplenames', help='Comma-separated list of sample names to include (or path to the csv file that lists them)')
    parser.add_argument('rawfile_top_dir',     help='Path to directory that holds each of the [samplename] directories that contain raw files')
    #optional arguments
    parser.add_argument('--layers',               default=[-1],           type=split_csv_to_list_of_ints,         
        help='Comma-separated list of image layer numbers to consider (default -1 means all layers)')
    parser.add_argument('--max_images',           default=-1,             type=int,         
        help='Number of images to load from the inputted list of samples')
    parser.add_argument('--selection',            default='random',       choices=['random','first','last'],
        help='Select "first", "last", or "random" n images (where n=max_images) from the inputted sample list. Default is "random".')
    parser.add_argument('--n_threads',            default=10,             type=int,         
        help='Number of threads to run at once in speeding up file I/O')
    parser.add_argument('--skip_masking', action='store_true',
        help='Add this flag to skip masking out the background regions of the images as they get added')
    parser.add_argument('--save_masking_plots', action='store_true',
        help='Add this flag to save the step-by-step plots of the masks produced as they are calculated')
    parser.add_argument('--rawfile_ext',          default='.Data.dat',
        help='Extension of raw files to load (default is ".Data.dat")')
    parser.add_argument('--workingdir_name',      default='flatfield_test',
        help='Name of working directory to save created files in')
    parser.add_argument('--flatfield_image_name', default='flatfield',
        help='Stem for meanimage file names')
    args = parser.parse_args()
    #make sure the command line arguments are valid
    checkArgs(args)
    #get the image file dimensions from the .xml file
    dims = getImageHWLFromXMLFile(args.rawfile_top_dir,sample_names[0])
    #get the list of filepaths to run and the names of their samples
    filepaths_to_run, sample_names_to_run = getFilepathsAndSampleNamesToRun(args)
    #start up a flatfield producer
    ff_producer = FlatfieldProducer(dims,filepaths_to_run,sample_names_to_run,args.n_threads)
    #begin by finding the background threshold per layer by looking at the HPFs on the tissue edges
    ff_producer.findBackgroundThreshold()
    #mask and stack images together
    ff_producer.stackImages()
    #make the flatfield image
    ff_producer.makeFlatField()
    #save the flatfield image and all the plots, etc.
    ff_producer.writeOutInfo()

if __name__=='__main__' :
    main()
    