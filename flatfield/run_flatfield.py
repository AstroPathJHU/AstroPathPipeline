#imports
from .flatfield_producer import FlatfieldProducer
from .utilities import flatfield_logger, sampleNameFromFilepath, FlatfieldSampleInfo
from .config import CONST 
from ..utilities.tableio import readtable
from ..utilities.misc import cd, split_csv_to_list, addCommonArgumentsToParser
from argparse import ArgumentParser
import os, glob, random, sys

#################### FILE-SCOPE CONSTANTS ####################

FILEPATH_TEXT_FILE_NAME = 'filepath_log.txt' #what the filepath log file is called

#################### HELPER FUNCTIONS ####################

#helper function to make sure arguments are valid
def checkArgs(a) :
    #if the user wants to apply a previously-calculated flatfield, the flatfield itself and rawfile log both have to exist in the prior run dir
    if a.mode=='apply_flatfield' :  
        if a.prior_run_dir is None :
            raise RuntimeError('ERROR: apply_flatfield mode requires a specified prior_run_dir!')
        if not os.path.isdir(a.prior_run_dir) :
            raise ValueError(f'ERROR: prior run directory {a.prior_run_dir} does not exist!')
        prior_run_ff_filename = f'{CONST.FLATFIELD_FILE_NAME_STEM}_{os.path.basename(os.path.normpath(a.prior_run_dir))}{CONST.FILE_EXT}'
        if not os.path.isfile(os.path.join(a.prior_run_dir,prior_run_ff_filename)) :
            raise ValueError(f'ERROR: flatfield image {prior_run_ff_filename} does not exist in prior run directory {a.prior_run_dir}!')
        if not os.path.isfile(os.path.join(a.prior_run_dir,f'{FILEPATH_TEXT_FILE_NAME}')) :
            raise ValueError(f'ERROR: rawfile log {FILEPATH_TEXT_FILE_NAME} does not exist in prior run directory {a.prior_run_dir}!')
    #read in the samples' information
    if not os.path.isfile(a.samples) :
        raise ValueError(f'ERROR: FlatfieldSampleInfo file {a.samples} does not exist!')
    samples = readtable(a.samples,FlatfieldSampleInfo)
    #make sure all of the samples' necessary directories exist
    for sample in samples :
        rfd = os.path.join(sample.rawfile_top_dir,sample.name)
        if not os.path.isdir(rfd) :
            raise ValueError(f'ERROR: rawfile directory {rfd} does not exist!')
        mfd = os.path.join(sample.metadata_top_dir,sample.name)
        if not os.path.isdir(rfd) :
            raise ValueError(f'ERROR: metadata file directory {mfd} does not exist!')
    #create the working directory if it doesn't already exist
    if not os.path.isdir(a.workingdir_name) :
        os.mkdir(a.workingdir_name)
    #if exposure time corrections are being done, make sure the file actually exists
    if not a.skip_exposure_time_correction :
        if not os.path.isfile(a.exposure_time_offset_file) :
            raise ValueError(f'ERROR: exposure time offset file {a.exposure_time_offset_file} does not exist!')
    #if the user wants to save example masking plots, they can't be skipping masking
    if a.skip_masking and a.n_masking_images_per_sample!=0 :
        raise RuntimeError("ERROR: can't save masking images if masking is being skipped!")
    #make sure the threshold file directory exists if it's specified
    if a.threshold_file_dir is not None and (not os.path.isdir(a.threshold_file_dir)) :
        raise ValueError(f'ERROR: Threshold file directory {a.threshold_file_dir} does not exist!')
    #make sure the selected pixel fraction is a valid number
    if a.selected_pixel_cut<0.0 or a.selected_pixel_cut>1.0 :
        raise ValueError(f'ERROR: selected pixel cut fraction {a.selected_pixel_cut} must be between 0 and 1!')
    #if the user wants to exclude other directories as well, their filepath logs all have to exist
    if a.other_runs_to_exclude!=[''] :
        for prd in a.other_runs_to_exclude :
            if not os.path.isfile(os.path.join(prd,f'{FILEPATH_TEXT_FILE_NAME}')) :
                raise ValueError(f'ERROR: raw file path log {FILEPATH_TEXT_FILE_NAME} does not exist in additional prior run directory {prd}!')

#helper function to get the list of filepaths and associated sample names to run on based on the selection method and number of images requested
def getFilepathsAndSamplesToRun(a) :
    #first we need to read in the inputted samples
    all_samples = readtable(a.samples,FlatfieldSampleInfo)
    samples_to_run = None; filepaths_to_run = None; filepaths_to_exclude = None
    #If other runs are being excluded, make sure to save their filenames to remove them
    if a.other_runs_to_exclude!=[''] :
        filepaths_to_exclude=[]
        for prd in a.other_runs_to_exclude :
            this_rawfile_log_path = os.path.join(prd,FILEPATH_TEXT_FILE_NAME)
            with open(this_rawfile_log_path,'r') as f:
                these_filepaths_to_exclude=[l.rstrip() for l in f.readlines() if l.rstrip()!='']
                flatfield_logger.info(f'Will exclude {len(these_filepaths_to_exclude)} files listed in previous run at {prd}')
                filepaths_to_exclude+=these_filepaths_to_exclude
    #If a prior run is specified, read the filepaths from there (either to run them again or exclude them)
    if a.prior_run_dir is not None :
        rawfile_log_path = os.path.join(a.prior_run_dir,FILEPATH_TEXT_FILE_NAME)
        with open(rawfile_log_path,'r') as f:
            previously_run_filepaths = [l.rstrip() for l in f.readlines() if l.rstrip()!='']
        #if those filepaths will be excluded, note that
        if a.mode=='apply_flatfield' or filepaths_to_exclude is not None:
            if filepaths_to_exclude is None :
                filepaths_to_exclude=[]
            filepaths_to_exclude+=previously_run_filepaths
            flatfield_logger.info(f'Will exclude {len(previously_run_filepaths)} files listed in previous run at {a.prior_run_dir}')
        #otherwise those filepaths are the ones to run; use them to find the subset of the samples to run
        else :
            filepaths_to_run = previously_run_filepaths
            flatfield_logger.info(f'Will run on a sample of {len(filepaths_to_run)} total files as listed in previous run at {a.prior_run_dir}')
            sample_names = list(set([sampleNameFromFilepath(fp) for fp in previously_run_filepaths]))
            samples_to_run = [s for s in all_samples if s.name in sample_names]
    #make a set of all the filepaths to exclude
    if filepaths_to_exclude is not None :
        filepaths_to_exclude=set(filepaths_to_exclude)
        flatfield_logger.info(f'{len(filepaths_to_exclude)} total files will be excluded from the samples')
    #if a prior run didn't define the samples to run, all the samples will be used
    if samples_to_run is None :
        samples_to_run = all_samples
    #get the sorted list of all rawfile paths in the samples that will be run
    all_sample_filepaths=[]
    for s in samples_to_run :
        with cd(os.path.join(s.rawfile_top_dir,s.name)) :
            all_sample_filepaths+=[os.path.join(s.rawfile_top_dir,s.name,fn) for fn in glob.glob(f'{s.name}_[[]*,*[]]{a.rawfile_ext}')]
    all_sample_filepaths.sort()
    #if the rawfiles haven't already been selected, figure that out
    if filepaths_to_run is None :
        #start with all of the files in every sample
        filepaths_to_run = all_sample_filepaths
        #remove filepaths to exclude
        if filepaths_to_exclude is not None :
            filepaths_to_run = [fp for fp in filepaths_to_run if fp not in filepaths_to_exclude]
        #select the filepaths to run on based on the criteria
        logstring='Will run using'
        if min(a.max_images,len(filepaths_to_run)) in [-1,len(filepaths_to_run)] :
            logstring+=f' all {len(filepaths_to_run)}'
            if filepaths_to_exclude is not None :
                logstring+=' remaining'
            logstring+=' images' 
            if min(a.max_images,len(filepaths_to_run)) > len(filepaths_to_run) :
                logstring+=f' (not enough images in the sample(s) to deliver all {a.max_images} requested)'
            filepaths_to_run = filepaths_to_run
        else :
            if a.selection_mode=='first' :
                filepaths_to_run=filepaths_to_run[:a.max_images]
                logstring+=f' the first {a.max_images} images'
            elif a.selection_mode=='last' :
                filepaths_to_run=filepaths_to_run[-(a.max_images):]
                logstring+=f' the last {a.max_images} images'
            elif a.selection_mode=='random' :
                random.shuffle(filepaths_to_run)
                filepaths_to_run=filepaths_to_run[:a.max_images]
                logstring+=f' {a.max_images} randomly-chosen images'
        #figure out the samples from which those files are coming and reset the list of all of those samples' filepaths
        samplenames_to_run = list(set([sampleNameFromFilepath(fp) for fp in filepaths_to_run]))
        samples_to_run = [s for s in samples_to_run if s.name in samplenames_to_run]
        logstring+=f' from {len(samples_to_run)} sample(s): '
        for s in samples_to_run :
            logstring+=f'{s.name}, '
        flatfield_logger.info(logstring[:-2])
        all_sample_filepaths = [fp for fp in all_sample_filepaths if sampleNameFromFilepath(fp) in [s.name for s in samples_to_run]]
    #alert the user if exposure time corrections will be made
    if a.skip_exposure_time_correction :
        flatfield_logger.info('Corrections for differences in exposure time will NOT be made')
    else :
        msg = 'Corrections for differences in exposure time will be made'
        msg+=f' based on the correction factors in {a.exposure_time_offset_file}'
        flatfield_logger.info(msg)
    #alert the user if masking will be applied
    if a.skip_masking :
        flatfield_logger.info('Images will NOT be masked before stacking')
    else :
        flatfield_logger.info('Images WILL be masked before stacking')
    #alert the user if the thresholds will be calculated in this run
    if (not a.skip_masking) and a.threshold_file_dir is None :
        logstring = f'Background thresholds will be calculated for {len(samples_to_run)}'
        if len(samples_to_run)>1 :
            logstring+=' different samples:'
        else :
            logstring+=' sample'
        logstring+=': '
        for s in samples_to_run :
            logstring+=f'{s.name}, '
        flatfield_logger.info(logstring[:-2])
    elif a.skip_masking :
        flatfield_logger.info('Background thresholds will not be calculated because masking will be skipped')
    elif a.threshold_file_dir is not None :
        flatfield_logger.info(f'Background thresholds will be read from the files at {a.threshold_file_dir}')
    #tell the user whether the tissue edge HPFs will be included
    if a.allow_edge_HPFs :
        flatfield_logger.info('HPFs on tissue edges will be included in the image stack.')
    else :
        flatfield_logger.info('HPFs on tissue edges will be found for each sample and excluded from the image stack.')
    #return the lists of filepaths and samplenames
    if len(all_sample_filepaths)<1 or len(filepaths_to_run)<1 or len(samples_to_run)<1 :
        raise RuntimeError('ERROR: The requested options have resulted in no samples or files to run!')
    return all_sample_filepaths, filepaths_to_run, samples_to_run

#################### MAIN SCRIPT ####################
def main() :
    #define and get the command-line arguments
    parser = ArgumentParser()
    #general positional arguments
    parser.add_argument('mode', choices=['make_flatfield','apply_flatfield','calculate_thresholds','check_run','choose_image_files'],                  
                        help='Which operation to perform')
    parser.add_argument('samples',
                        help='Path to .csv file listing FlatfieldSampleInfo objects to use samples from multiple raw/metadata file paths')
    parser.add_argument('workingdir_name', 
                        help='Name of working directory to save created files in')
    #add the exposure time correction group to the arguments
    addCommonArgumentsToParser(parser,positional_args=False,flatfielding=False,warping=False)
    #mutually exclusive group for how to handle the thresholding
    thresholding_group = parser.add_mutually_exclusive_group()
    thresholding_group.add_argument('--threshold_file_dir',
                                    help="""Path to the directory holding background threshold files created in previous runs for the samples in question
                                    [use this argument to re-use previously-calculated background thresholds]""")
    thresholding_group.add_argument('--skip_masking',       action='store_true',
                                    help="""Add this flag to entirely skip masking out the background regions of the images as they get added
                                    [use this argument to completely skip the background thresholding and masking]""")
    #group for how to select a subset of the samples' files
    file_selection_group = parser.add_argument_group('file selection',
                                                     'how many images from the sample set should be used, how to choose them, and where to find them')
    file_selection_group.add_argument('--prior_run_dir',     
                                      help="""Path to the working directory of a previous run whose raw files you want to use again, or whose calculated
                                      flatfield you want to apply to a different, orthogonal, set of files in the same samples""")
    file_selection_group.add_argument('--rawfile_ext',     default='.Data.dat',
                                      help='Extension of raw files to load (default is ".Data.dat")')
    file_selection_group.add_argument('--max_images',      default=-1,       type=int,         
                                      help='Number of images to load from the inputted list of samples')
    file_selection_group.add_argument('--selection_mode',  default='random', choices=['random','first','last'],
                                      help='Select "first", "last", or "random" (default) n images (where n=max_images) from the sample group.')
    file_selection_group.add_argument('--allow_edge_HPFs', action='store_true',
                                      help="""Add this flag to allow HPFs on the tissue edges to be stacked (not allowed by default)""")
    #group for some run options
    run_option_group = parser.add_argument_group('run options','other options for this run')
    run_option_group.add_argument('--n_threads',                   default=10,  type=int,         
                                  help='Number of threads/processes to run at once in parallelized portions of the code')
    run_option_group.add_argument('--n_masking_images_per_sample', default=0,   type=int,         
                                  help='How many example masking images to save for each sample (randomly chosen)')
    run_option_group.add_argument('--selected_pixel_cut',          default=0.8, type=float,         
                                  help='Minimum fraction (0->1) of pixels that must be selected as signal for an image to be added to the stack')
    run_option_group.add_argument('--other_runs_to_exclude',       default='',  type=split_csv_to_list,
                                  help='Comma-separated list of additional, previously-run, working directories whose filepaths should be excluded')
    args = parser.parse_args()
    #make sure the command line arguments make sense
    checkArgs(args)
    #get the list of filepaths to run and the names of their samples
    all_filepaths, filepaths_to_run, samples_to_run = getFilepathsAndSamplesToRun(args)
    if args.mode=='check_run' :
        sys.exit()
    #start up a flatfield producer
    ff_producer = FlatfieldProducer(samples_to_run,filepaths_to_run,args.workingdir_name,args.skip_exposure_time_correction,args.skip_masking)
    #write out the text file of all the raw file paths that will be run
    ff_producer.writeFileLog(FILEPATH_TEXT_FILE_NAME)
    if args.mode=='choose_image_files' :
        sys.exit()
    #First read in the exposure time correction offsets from the given directory
    if not args.skip_exposure_time_correction :
        ff_producer.readInExposureTimeCorrectionOffsets(args.exposure_time_offset_file)
    #next figure out the background thresholds per layer by looking at the HPFs on the tissue edges
    if not args.skip_masking :
        if args.threshold_file_dir is not None :
            ff_producer.readInBackgroundThresholds(args.threshold_file_dir)
        else :
            ff_producer.findBackgroundThresholds(all_filepaths,args.n_threads)
    if args.mode in ['make_flatfield', 'apply_flatfield'] :
        #mask and stack images together
        ff_producer.stackImages(args.n_threads,args.selected_pixel_cut,args.n_masking_images_per_sample,args.allow_edge_HPFs)
        if args.mode=='make_flatfield' :
            #make the flatfield image
            ff_producer.makeFlatField()
        if args.mode=='apply_flatfield' :
            #apply the flatfield to the image stack
            prior_run_ff_filename = f'{CONST.FLATFIELD_FILE_NAME_STEM}_{os.path.basename(os.path.normpath(args.prior_run_dir))}{CONST.FILE_EXT}'
            ff_producer.applyFlatField(os.path.join(args.prior_run_dir,prior_run_ff_filename))
        #save the plots, etc.
        ff_producer.writeOutInfo()
    flatfield_logger.info('All Done!')

if __name__=='__main__' :
    main()
    