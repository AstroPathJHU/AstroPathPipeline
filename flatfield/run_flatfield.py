#imports
from .flatfield_producer import FlatfieldProducer
from .config import *
from ..utilities.img_file_io import getImageHWLFromXMLFile
from ..utilities.misc import split_csv_to_list
from .utilities import sampleNameFromFilepath
from argparse import ArgumentParser
import shutil, glob, random, sys

#################### HELPER FUNCTIONS ####################

#helper function to make sure arguments are valid
def checkArgs(a) :
    #make sure any specified directory paths exist
    dirpath_args = [a.rawfile_prior_run_dir,a.dbload_top_dir,a.threshold_file_dir,a.rawfile_top_dir]
    for dp in dirpath_args :
        if dp is not None and not os.path.isdir(dp) :
            raise RuntimeError(f'ERROR: Directory {dp} does not exist!')
    #if the user is giving a list of samples, they must also specify where the rawfiles are
    if a.sample_names is not None and a.rawfile_top_dir is None :
        raise RuntimeError('ERROR: If sample names are given a rawfile location must also be specified through --rawfile_top_dir!')
    #if the user wants to calculate the thresholds then they need to supply the dbload directory
    if a.mode=='calculate_thresholds' and a.dbload_top_dir is None :
        raise RuntimeError('ERROR: calculating background thresholds needs a dbload directory location specified through --dbload_top_dir!')
    #create/replace the working directory if it doesn't already exist
    if os.path.isdir(a.workingdir_name) :
        flatfield_logger.info(f'Deleting previous run directory with name {a.workingdir_name}...')
        shutil.rmtree(a.workingdir_name)
        flatfield_logger.info(f'Done.')
    os.mkdir(a.workingdir_name)

#helper function to get the list of filepaths and associated sample names to run on based on the selection method and number of images requested
def getFilepathsAndSampleNamesToRun(a) :
    #first we need to read the user's input of either a previous rawfile log or a list of files in an argument or file
    all_sample_names = None; filepaths_to_run = None
    #If a prior run is being duplicated, then just read the files from there
    if a.rawfile_prior_run_dir is not None :
        rawfile_log_path = os.path.join(a.rawfile_prior_run_dir,FILEPATH_TEXT_FILE_NAME)
        if not os.path.isfile(rawfile_log_path) :
            raise ValueError(f'ERROR: previous run rawfile log {rawfile_log_path} does not exist!')
        with open(rawfile_log_path,'r') as f:
            filepaths_to_run = [l.rstrip() for l in f.readlines() if l.rstrip()!='']
        flatfield_logger.info(f'Will run on a sample of {len(filepaths_to_run)} total files as listed in previous run at {a.rawfile_prior_run_dir}')
        #use the filepaths to figure out the sample names
        all_sample_names = samplenames_to_run = list(set([sampleNameFromFilepath(fp) for fp in filepaths_to_run]))
    #Otherwise the sample names are defined instead and we need to select rawfiles from them
    elif a.sample_names is not None :
        #if they're defined in a file read them from there
        if '.txt' in a.sample_names :
            if not os.path.isfile(a.sample_names) :
                raise ValueError(f'ERROR: sample name text file {a.sample_names} does not exist!')
            #get the file contents
            with open(a.sample_names,'r') as f:
                all_sample_names = [l.rstrip() for l in f.readlines() if l.rstrip()!='']
        #otherwise they should just be listed in the command line argument
        else :
            all_sample_names = split_csv_to_list(a.sample_names)
    #get the filepaths of all the images in the samples to run if we need to choose from them or use them for the background thresholding
    all_sample_image_filepaths=[]
    #figure out where the files are located
    rawfile_top_dir=None
    if a.rawfile_top_dir is not None :
        rawfile_top_dir = a.rawfile_top_dir
    else :
        fpsplit = filepaths_to_run[0].split(os.sep)
        rawfile_top_dir = os.path.join(*[fpp for fpp in fpsplit[:fpsplit.index(sampleNameFromFilepath(filepaths_to_run[0]))]])
    #make sure the rawfile (and dbload, if thresholds will be calculated) directories all exist first
    will_calculate_thresholds = (not a.skip_masking) and a.threshold_file_dir is None
    for sn in all_sample_names :
        if not os.path.isdir(os.path.join(rawfile_top_dir,sn)) :
            raise ValueError(f'ERROR: sample directory {os.path.join(rawfile_top_dir,sn)} does not exist!')
        if will_calculate_thresholds :
            if not os.path.isdir(os.path.join(a.dbload_top_dir,sn,'dbload')) :
                raise ValueError(f"ERROR: dbload directory {os.path.join(a.dbload_top_dir,sn,'dbload')} for sample {sn} does not exist!")
    #get the (sorted) full list of file names in each sample to choose from
    for sn in all_sample_names :
        #get the list of all the filenames for this sample
        with cd(os.path.join(rawfile_top_dir,sn)) :
            this_sample_image_filepaths = [os.path.join(rawfile_top_dir,sn,fn) 
                                            for fn in glob.glob(f'{sn}_[[]*,*[]]{a.rawfile_ext}')]
            all_sample_image_filepaths+=this_sample_image_filepaths
    all_sample_image_filepaths.sort()
    #if the rawfiles haven't already been selected, figure that out
    if filepaths_to_run is None :
        #select the total filepaths/sample names to run on
        logstring='Will run using'
        if min(a.max_images,len(all_sample_image_filepaths)) in [-1,len(all_sample_image_filepaths)] :
            logstring+=f' all {len(all_sample_image_filepaths)} images' 
            if min(a.max_images,len(all_sample_image_filepaths)) == len(all_sample_image_filepaths) :
                logstring+=f' (not enough images in the sample(s) to deliver all {a.max_images} requested)'
            filepaths_to_run = all_sample_image_filepaths
        else :
            if a.selection_mode=='first' :
                filepaths_to_run=all_sample_image_filepaths[:a.max_images]
                logstring+=f' the first {a.max_images} images'
            elif a.selection_mode=='last' :
                filepaths_to_run=all_sample_image_filepaths[-(a.max_images):]
                logstring+=f' the last {a.max_images} images'
            elif a.selection_mode=='random' :
                random.shuffle(all_sample_image_filepaths)
                filepaths_to_run=all_sample_image_filepaths[:a.max_images]
                logstring+=f' {a.max_images} randomly-chosen images'
        #figure out the samples from which those files are coming
        samplenames_to_run = list(set([sampleNameFromFilepath(fp) for fp in filepaths_to_run]))
        logstring+=f' from {len(samplenames_to_run)} sample(s): '
        for sn in samplenames_to_run :
            logstring+=f'{sn}, '
        flatfield_logger.info(logstring[:-2])
    if will_calculate_thresholds :
        logstring = f'Background thresholds will be calculated for {len(samplenames_to_run)}'
        if len(samplenames_to_run)>1 :
            logstring+=' different samples:'
        else :
            logstring+=' sample'
        logstring+=': '
        for sn in samplenames_to_run :
            logstring+=f'{sn}, '
        flatfield_logger.info(logstring[:-2])
    elif a.skip_masking :
        flatfield_logger.info('Background thresholds will not be calculated since masking will be skipped')
    elif a.threshold_file_dir is not None :
        flatfield_logger.info(f'Background thresholds will be read from the files at {a.threshold_file_dir}')
    #return the lists of filepaths and samplenames
    all_filepaths = [fp for fp in all_sample_image_filepaths if sampleNameFromFilepath(fp) in samplenames_to_run]
    return all_filepaths, filepaths_to_run, samplenames_to_run

#################### MAIN SCRIPT ####################
def main() :
    #define and get the command-line arguments
    parser = ArgumentParser()
    #general positional arguments
    parser.add_argument('mode', default='make_flatfield', choices=['make_flatfield','calculate_thresholds','visualize_masking','check_run'],                  
                        help='Which operation to perform')
    parser.add_argument('workingdir_name', 
                        help='Name of working directory to save created files in')
    #mutually exclusive group for how to specify the samples that will be used
    samplenames_group = parser.add_mutually_exclusive_group(required=True)
    samplenames_group.add_argument('--sample_names',     
                                   help="""Comma-separated list of sample names to use, or path to a file that lists them line by line 
                                   [use this argument if this run is using a new set of samples]""")
    samplenames_group.add_argument('--rawfile_prior_run_dir',     
                                   help="""Path to the working directory of a previous run whose raw files you want to use again 
                                   [use this argument instead of defining a new set of samples]""")
    #mutually exclusive group for how to handle the thresholding
    thresholding_group = parser.add_mutually_exclusive_group(required=True)
    thresholding_group.add_argument('--dbload_top_dir',  
                                    help="""Path to directory containing each of the [samplename]/dbload directories with *.csv files in them
                                    [use this argument to calculate the background thresholds from scratch]""")
    thresholding_group.add_argument('--threshold_file_dir',
                                    help="""Path to the directory holding background threshold files created in previous runs for the samples in question
                                    [use this argument to re-use previously-calculated background thresholds]""")
    thresholding_group.add_argument('--skip_masking',       action='store_true',
                                    help="""Add this flag to entirely skip masking out the background regions of the images as they get added
                                    [use this argument to completely skip the background thresholding and masking]""")
    #group for how to select a subset of the samples' files
    file_selection_group = parser.add_argument_group('file selection',
                                                     'how many images from the sample set should be used, how to choose them, and where to find them')
    file_selection_group.add_argument('--rawfile_top_dir',
                                      help='Path to directory containing each of the [samplename] directories with raw files in them')
    file_selection_group.add_argument('--rawfile_ext',    default='.Data.dat',
                                      help='Extension of raw files to load (default is ".Data.dat")')
    file_selection_group.add_argument('--max_images',     default=-1,       type=int,         
                                      help='Number of images to load from the inputted list of samples')
    file_selection_group.add_argument('--selection_mode', default='random', choices=['random','first','last'],
                                      help='Select "first", "last", or "random" (default) n images (where n=max_images) from the sample group.')
    #group for some run options
    run_option_group = parser.add_argument_group('run options','other options for this run')
    run_option_group.add_argument('--n_threads',       default=10,    type=int,         
                                  help='Number of threads/processes to run at once in parallelized portions of the code')
    args = parser.parse_args()
    #make sure the command line arguments make sense
    checkArgs(args)
    #get the list of filepaths to run and the names of their samples
    all_filepaths, filepaths_to_run, sample_names_to_run = getFilepathsAndSampleNamesToRun(args)
    if args.mode=='check_run' :
        sys.exit()
    #get the image file dimensions from the .xml file
    dims = getImageHWLFromXMLFile(args.rawfile_top_dir,sample_names_to_run[0])
    #start up a flatfield producer
    ff_producer = FlatfieldProducer(dims,sample_names_to_run,args.workingdir_name,args.skip_masking)
    #begin by figuring out the background thresholds per layer by looking at the HPFs on the tissue edges
    if not args.skip_masking :
        if args.threshold_file_dir is not None :
            ff_producer.readInBackgroundThresholds(args.threshold_file_dir)
        elif args.dbload_top_dir is not None :
            ff_producer.findBackgroundThresholds(all_filepaths,args.dbload_top_dir,args.n_threads)
    if args.mode in ['make_flatfield','visualize_masking'] :
        #mask and stack images together
        ff_producer.stackImages(filepaths_to_run,args.n_threads,args.mode=='visualize_masking')
        if args.mode=='make_flatfield' :
            #make the flatfield image
            ff_producer.makeFlatField()
            #save the flatfield image and all the plots, etc.
            ff_producer.writeOutInfo()
    flatfield_logger.info('All Done!')

if __name__=='__main__' :
    main()
    