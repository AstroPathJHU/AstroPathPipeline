#imports
from .flatfield_producer import FlatfieldProducer
from .config import *
from ..utilities.img_file_io import getImageHWLFromXMLFile
from ..utilities.misc import cd, split_csv_to_list
from .utilities import sampleNameFromFilepath
from argparse import ArgumentParser
import os, glob, random

#################### HELPER FUNCTIONS ####################

#helper function to make sure arguments are valid
def checkArgs(a) :
    #make sure that the raw file directory exists
    if not os.path.isdir(a.rawfile_top_dir) :
        raise ValueError(f'ERROR: Raw file directory {a.rawfile_top_dir} does not exist!')
    #make sure that the dbload top directory exists
    if not os.path.isdir(a.dbload_top_dir) :
        raise ValueError(f'ERROR: dbload top directory {a.dbload_top_dir} does not exist!')
    #create the working directory if it doesn't already exist
    if not os.path.isdir(a.workingdir_name) :
        os.mkdir(a.workingdir_name)
    #make sure that the user didn't ask to save the masking plots while skipping the masking
    if a.skip_masking and a.save_masking_plots :
        raise ValueError('ERROR: cannot save masking plots if masking will be skipped')
    #make sure that the user didn't ask to just determine the thresholds while skipping the masking
    if a.threshold_only and a.skip_masking :
        raise ValueError('ERROR: cannot skip masking while running to determine background thresholds')
    #make sure the threshold file directory exists if it's going to be used
    if a.threshold_file_dir is not None and not os.path.isdir(a.threshold_file_dir) :
        raise ValueError(f'ERROR: threshold file directory {a.threshold_file_dir} does not exist!')

#helper function to get the list of filepaths and associated sample names to run on based on the selection method and number of images requested
def getFilepathsAndSampleNamesToRun(a) :
    samplenames_to_run = None; filepaths_to_run = None
    #get sample names/filepaths from the .csv file if requested
    previous_log_exists = os.path.isfile(os.path.join(a.samplenames,FILEPATH_TEXT_FILE_NAME))
    if previous_log_exists or '.txt' in a.samplenames :
        filepath = os.path.join(a.samplenames,FILEPATH_TEXT_FILE_NAME) if previous_log_exists else a.samplenames
        #make sure that the text file exists
        if not os.path.isfile(filepath) :
            raise ValueError(f'ERROR: file/sample name text file {a.samplenames} does not exist!')
        #get the file contents
        with open(filepath,'r') as f:
            file_lines = [l.rstrip() for l in f.readlines()]
        #if the input file was filepaths, then each entry should have the rawfile directory in it
        if a.rawfile_top_dir.split(os.sep)[-1] in file_lines[0].split(os.sep) :
            filepaths_to_run = file_lines
            flatfield_logger.info(f'Will run on a sample of {len(filepaths_to_run)} total files as listed in {a.samplenames}')
            samplenames_to_run = list(set([sampleNameFromFilepath(fp) for fp in filepaths_to_run]))
            all_sample_names = samplenames_to_run
        #otherwise the input file is sample names
        else :
            all_sample_names = file_lines
    #if the sample/filenames weren't given in a .csv the samples should just be listed in the command line argument
    else :
        all_sample_names = split_csv_to_list(a.samplenames)
    #get the filepaths of all the images in the samples to run, regardless of how the files to run on will themselves be chosen
    all_image_filepaths=[]
    #make sure the rawfile and dbload directories all exist first
    for sn in all_sample_names :
        if not os.path.isdir(os.path.join(a.rawfile_top_dir,sn)) :
            raise ValueError(f'ERROR: sample directory {os.path.join(a.rawfile_top_dir,sn)} does not exist!')
        if not os.path.isdir(os.path.join(a.dbload_top_dir,sn,'dbload')) :
            raise ValueError(f"ERROR: dbload directory {os.path.join(a.dbload_top_dir,sn,'dbload')} for sample {sn} does not exist!")
    #get the (sorted) full list of file names in each sample to choose from
    for sn in all_sample_names :
        #get the list of all the filenames for this sample
        with cd(os.path.join(a.rawfile_top_dir,sn)) :
            this_sample_image_filepaths = [os.path.join(a.rawfile_top_dir,sn,fn) 
                                            for fn in glob.glob(f'{sn}_[[]*,*[]]{a.rawfile_ext}')]
            all_image_filepaths+=this_sample_image_filepaths
    all_image_filepaths.sort()
    #if the rawfiles haven't already been selected, figure that out
    if filepaths_to_run is None :
        #select the total filepaths/sample names to run on
        logstring='Will run on a sample of'
        if min(a.max_images,len(all_image_filepaths)) in [-1,len(all_image_filepaths)] :
            logstring+=f' all {len(all_image_filepaths)} images' 
            if min(a.max_images,len(all_image_filepaths)) == len(all_image_filepaths) :
                logstring+=f' (not enough images in the sample(s) to deliver all {a.max_images} requested)'
            filepaths_to_run = all_image_filepaths
        else :
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
        samplenames_to_run = list(set([sampleNameFromFilepath(fp) for fp in filepaths_to_run]))
    logstring = f'Background threshold will be determined from images in {len(samplenames_to_run)}'
    if len(samplenames_to_run)>1 :
        logstring+=' different samples:'
    else :
        logstring+=' sample'
    logstring+=': '
    for sn in samplenames_to_run :
        logstring+=f'{sn}, '
    flatfield_logger.info(logstring[:-2])
    #return the lists of filepaths and samplenames
    all_filepaths = [fp for fp in all_image_filepaths if sampleNameFromFilepath(fp) in samplenames_to_run]
    return all_filepaths, filepaths_to_run, samplenames_to_run

#################### MAIN SCRIPT ####################
def main() :
    #define and get the command-line arguments
    parser = ArgumentParser()
    #positional arguments
    parser.add_argument('samplenames',     help="""List of sample names to include (or path to the file that lists either 
                                                   sample names or the files from a previous run, one per line)""")
    parser.add_argument('rawfile_top_dir', help='Path to directory that holds each of the [samplename] directories that contain raw files')
    parser.add_argument('dbload_top_dir',  help="""Path to directory that holds each of the [samplename]/dbload directories 
                                                   which contain *_overlap.csv and *_rect.csv files""")
    #optional arguments
    parser.add_argument('--max_images',           default=-1,             type=int,         
        help='Number of images to load from the inputted list of samples')
    parser.add_argument('--selection',            default='random',       choices=['random','first','last'],
        help='Select "first", "last", or "random" n images (where n=max_images) from the inputted sample list. Default is "random".')
    parser.add_argument('--n_threads',            default=10,             type=int,         
        help='Number of threads to run at once in speeding up file I/O')
    parser.add_argument('--threshold_file_dir',   default=None,
        help='Path to the directory holding background threshold files created in previous runs for the samples in question')
    parser.add_argument('--threshold_only',       action='store_true',
        help='Add this flag to exit after determining and saving the background thresholds for the given samples')
    parser.add_argument('--skip_masking',         action='store_true',
        help='Add this flag to skip masking out the background regions of the images as they get added')
    parser.add_argument('--save_masking_plots',   action='store_true',
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
    #get the list of filepaths to run and the names of their samples
    all_filepaths, filepaths_to_run, sample_names_to_run = getFilepathsAndSampleNamesToRun(args)
    #get the image file dimensions from the .xml file
    dims = getImageHWLFromXMLFile(args.rawfile_top_dir,sample_names_to_run[0])
    #start up a flatfield producer
    ff_producer = FlatfieldProducer(dims,sample_names_to_run,args.workingdir_name,args.skip_masking)
    #begin by finding the background threshold per layer by looking at the HPFs on the tissue edges
    ff_producer.findBackgroundThresholds(all_filepaths,args.dbload_top_dir,args.n_threads,args.threshold_file_dir)
    if not args.threshold_only :
        #mask and stack images together
        ff_producer.stackImages(filepaths_to_run,args.n_threads,args.save_masking_plots)
        #make the flatfield image
        ff_producer.makeFlatField()
        #save the flatfield image and all the plots, etc.
        ff_producer.writeOutInfo(args.flatfield_image_name)
    flatfield_logger.info('All Done!')

if __name__=='__main__' :
    main()
    