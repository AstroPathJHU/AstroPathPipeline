#imports
from .flatfield_producer import FlatfieldProducer
from .logging import RunLogger
from .utilities import flatfield_logger, slideNameFromFilepath, FlatfieldSlideInfo
from .utilities import getSlideMeanImageWorkingDirPath, getBatchFlatfieldWorkingDirPath, getSlideMeanImageFilepath, getSlideMaskStackFilepath
from .config import CONST 
from ...utilities.tableio import readtable
from ...utilities.misc import cd, split_csv_to_list
from ...utilities.config import CONST as UNIV_CONST
from argparse import ArgumentParser
import os, glob, random, shutil

#################### FILE-SCOPE CONSTANTS ####################

FILEPATH_TEXT_FILE_NAME = 'filepath_log.txt' #what the filepath log file is called
DEFAULT_SELECTED_PIXEL_CUT = 0.8

#################### HELPER FUNCTIONS ####################

#helper function to make sure arguments are valid
def checkArgs(a) :
    #first check the different batch mode options, which don't accept every argument
    if a.mode=='slide_mean_image' or a.mode=='batch_flatfield' :
        #working directory name is automatic
        if a.workingdir is not None :
            raise ValueError(f'ERROR: running in {a.mode} mode uses an automatic working directory, please remove the workingdir argument!') 
        #need rawfile and root directories (i.e. don't use a sample definition csv file)
        if a.rawfile_top_dir is None or a.root_dir is None :
            raise ValueError(f'ERROR: must specify rawfile and root directories to run in {a.mode} mode!')
        #the whole file selection group is irrelevant (plus also other runs to exclude)
        if (a.prior_run_dir is not None) or (a.max_images!=-1) or (a.selection_mode!='random') or a.allow_edge_HPFs or a.other_runs_to_exclude!=[''] :
            raise ValueError(f'ERROR: file selection is done automatically in {a.mode} mode!')
        #only ever run using image masks
        if a.skip_masking :
            raise ValueError(f'ERROR: running in {a.mode} mode is not compatible with skipping masking! Remove the skip_masking flag!')
        #can't save masking images
        if a.n_masking_images_per_slide!=0 :
            raise ValueError(f'ERROR: cannot save masking images when running in {a.mode} mode!')
        #can only use the default selected pixel cut
        if a.selected_pixel_cut!=DEFAULT_SELECTED_PIXEL_CUT :
            raise ValueError(f'ERROR: when running in {a.mode} mode only the default selected pixel cut of {DEFAULT_SELECTED_PIXEL_CUT} is valid!')
        if a.mode=='slide_mean_image' :
            #need to apply exposure time corrections
            if a.exposure_time_offset_file is None :
                raise ValueError('ERROR: must give an exposure time offset file in slide_mean_image mode')
            #make sure it'll run on exactly one slide
            if len(split_csv_to_list(a.slides))!=1 :
                raise ValueError(f'ERROR: running in slide_mean_image mode requires running one slide at a time, but slides argument = {a.slides}!')
        elif a.mode=='batch_flatfield' :
            #explicitly NEED a batch ID
            if a.batchID==-1 :
                raise ValueError('ERROR: batchID argument is REQUIRED when running in batch_flatfield mode')
            #it has to be between 0 and 99
            if a.batchID<0 or a.batchID>99 :
                raise ValueError('ERROR: only batch IDs between 0 and 99 are supported')
            #shouldn't use exposure time arguments
            if a.skip_exposure_time_correction or a.exposure_time_offset_file is not None :
                raise ValueError('ERROR: exposure time arguments are irrelevant in batch_flatfield mode!')
    #otherwise there needs to be a working directory
    elif a.workingdir is None :
        raise ValueError('ERROR: the workingdir argument is required!') 
    #batchID not needed
    if a.mode!='batch_flatfield' and (a.batchID!=-1) :
        raise ValueError('ERROR: batchID argument is only valid in batch_flatfield mode!')
    #every mode except for batch_flatfield needs exposure time correction arguments
    if a.mode!='batch_flatfield' :
        if (not a.skip_exposure_time_correction) and a.exposure_time_offset_file is None :
            raise ValueError('ERROR: must either skip exposure time correction or give an offset file in every run mode except for batch_flatfield')
        #if exposure time corrections are being done, make sure the file actually exists
        elif not a.skip_exposure_time_correction :
            if not os.path.isfile(a.exposure_time_offset_file) :
                raise ValueError(f'ERROR: exposure time offset file {a.exposure_time_offset_file} does not exist!')
    #if the user wants to apply a previously-calculated flatfield, the flatfield itself and rawfile log both have to exist in the prior run dir
    if a.mode=='apply_flatfield' :  
        if a.prior_run_dir is None :
            raise RuntimeError('ERROR: apply_flatfield mode requires a specified prior_run_dir!')
        if not os.path.isdir(a.prior_run_dir) :
            raise ValueError(f'ERROR: prior run directory {a.prior_run_dir} does not exist!')
        with cd(a.prior_run_dir) :
            prior_run_ff_filename = glob.glob(f'*{CONST.FLATFIELD_FILE_NAME_STEM}*{CONST.FILE_EXT}')
        if len(prior_run_ff_filename)!=1 :
            raise ValueError(f'ERROR: {len(prior_run_ff_filename)} flatfield files to apply found in {a.prior_run_dir}')
        prior_run_ff_filename = prior_run_ff_filename[0]
        if not os.path.isfile(os.path.join(a.prior_run_dir,prior_run_ff_filename)) :
            raise ValueError(f'ERROR: flatfield image {prior_run_ff_filename} does not exist in prior run directory {a.prior_run_dir}!')
        if not os.path.isfile(os.path.join(a.prior_run_dir,f'{FILEPATH_TEXT_FILE_NAME}')) :
            raise ValueError(f'ERROR: rawfile log {FILEPATH_TEXT_FILE_NAME} does not exist in prior run directory {a.prior_run_dir}!')
    #figure out and read in the slides' information
    slides = None
    if a.slides is None :
        raise ValueError('ERROR: "slides" argument is required!')
    if '.csv' in a.slides :
        if (a.rawfile_top_dir is not None) or (a.root_dir is not None) :
            raise ValueError(f'ERROR: slides argument {a.slides} is a file, so rawfile/root_dir arguments are ambiguous!')
        try:
            if os.path.isfile(a.slides) :
                slides = readtable(a.slides,FlatfieldSlideInfo)
        except FileNotFoundError :
            raise ValueError(f'Slides file {a.slides} does not exist!')
    else :
        if (a.rawfile_top_dir is None) or (a.root_dir is None) :
            raise ValueError(f'ERROR: slides argument {a.slides} is not a file, but rawfile/root_dir arguments are not specified!')
        slides = []
        for sn in split_csv_to_list(a.slides) :
            slides.append(FlatfieldSlideInfo(sn,a.rawfile_top_dir,a.root_dir))
    if slides is None or len(slides)<1 :
        raise ValueError(f"""ERROR: Slides '{a.slides}', rawfile top dir '{a.rawfile_top_dir}', and root dir '{a.root_dir}' 
                             did not result in a valid list of slides!""")
    #make sure all of the slides' necessary directories exist
    for slide in slides :
        rfd = os.path.join(slide.rawfile_top_dir,slide.name)
        if not os.path.isdir(rfd) :
            raise ValueError(f'ERROR: rawfile directory {rfd} does not exist!')
        mfd = os.path.join(slide.root_dir,slide.name)
        if not os.path.isdir(rfd) :
            raise ValueError(f'ERROR: slide root directory {mfd} does not exist!')
        if a.mode=='batch_flatfield' :
            mi = getSlideMeanImageFilepath(slide)
            if not os.path.isfile(mi) :
                raise FileNotFoundError(f'ERROR: Required mean image file {mi} does not exist!')
            ms = getSlideMaskStackFilepath(slide)
            if not os.path.isfile(ms) :
                raise FileNotFoundError(f'ERROR: Required mask stack file {ms} does not exist!')
    #if the user wants to save example masking plots, they can't be skipping masking
    if a.skip_masking and a.n_masking_images_per_slide!=0 :
        raise RuntimeError("ERROR: can't save masking images if masking is being skipped!")
    #make sure the threshold file directory exists if it's specified, with the requisite sample background threshold files
    if a.threshold_file_dir is not None :
        if not os.path.isdir(a.threshold_file_dir) :
            raise ValueError(f'ERROR: Threshold file directory {a.threshold_file_dir} does not exist!')
        for sn in [s.name for s in slides] :
            tfn = f'{sn}_{UNIV_CONST.BACKGROUND_THRESHOLD_TEXT_FILE_NAME_STEM}'
            tfp = os.path.join(a.threshold_file_dir,tfn)
            if not os.path.isfile(tfp) :
                raise FileNotFoundError(f'ERROR: background threshold file {tfp} does not exist!')
    #make sure the selected pixel fraction is a valid number
    if a.selected_pixel_cut<0.0 or a.selected_pixel_cut>1.0 :
        raise ValueError(f'ERROR: selected pixel cut fraction {a.selected_pixel_cut} must be between 0 and 1!')
    #if the user wants to exclude other directories as well, their filepath logs all have to exist
    if a.other_runs_to_exclude!=[''] :
        for prd in a.other_runs_to_exclude :
            if not os.path.isfile(os.path.join(prd,f'{FILEPATH_TEXT_FILE_NAME}')) :
                raise ValueError(f'ERROR: raw file path log {FILEPATH_TEXT_FILE_NAME} does not exist in additional prior run directory {prd}!')

#helper function to get the list of filepaths and associated slide names to run on based on the selection method and number of images requested
def getFilepathsAndSlidesToRun(a,logger=None) :
    if logger is None :
        logger=flatfield_logger
    #first we need to read in the inputted slides
    if '.csv' in a.slides :
        all_slides = readtable(a.slides,FlatfieldSlideInfo)
    else :
        all_slides = []
        for sn in split_csv_to_list(a.slides) :
            all_slides.append(FlatfieldSlideInfo(sn,a.rawfile_top_dir,a.root_dir))
    slides_to_run = None; filepaths_to_run = None; filepaths_to_exclude = None
    #If other runs are being excluded, make sure to save their filenames to remove them
    if a.other_runs_to_exclude!=[''] :
        filepaths_to_exclude=[]
        for prd in a.other_runs_to_exclude :
            this_rawfile_log_path = os.path.join(prd,FILEPATH_TEXT_FILE_NAME)
            with open(this_rawfile_log_path,'r') as f:
                these_filepaths_to_exclude=[l.rstrip() for l in f.readlines() if l.rstrip()!='']
                logger.info(f'Will exclude {len(these_filepaths_to_exclude)} files listed in previous run at {prd}')
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
            logger.info(f'Will exclude {len(previously_run_filepaths)} files listed in previous run at {a.prior_run_dir}')
        #otherwise those filepaths are the ones to run; use them to find the subset of the slides to run
        else :
            filepaths_to_run = previously_run_filepaths
            logger.info(f'Will run on {len(filepaths_to_run)} total files as listed in previous run at {a.prior_run_dir}')
            slide_names = list(set([slideNameFromFilepath(fp) for fp in previously_run_filepaths]))
            slides_to_run = [s for s in all_slides if s.name in slide_names]
    #make a set of all the filepaths to exclude
    if filepaths_to_exclude is not None :
        filepaths_to_exclude=set(filepaths_to_exclude)
        logger.info(f'{len(filepaths_to_exclude)} total files will be excluded from the slides')
    #if a prior run didn't define the slides to run, all the slides will be used
    if slides_to_run is None :
        slides_to_run = all_slides
    #get the sorted list of all rawfile paths in the slides that will be run
    all_slide_filepaths=[]
    for s in slides_to_run :
        with cd(os.path.join(s.rawfile_top_dir,s.name)) :
            all_slide_filepaths+=[os.path.join(s.rawfile_top_dir,s.name,fn) for fn in glob.glob(f'{s.name}_[[]*,*[]]{UNIV_CONST.RAW_EXT}')]
    all_slide_filepaths.sort()
    #if the rawfiles haven't already been selected, figure that out
    if filepaths_to_run is None :
        #start with all of the files in every slide
        filepaths_to_run = all_slide_filepaths
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
                logstring+=f' (not enough images in the slide(s) to deliver all {a.max_images} requested)'
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
        #figure out the slides from which those files are coming and reset the list of all of those slides' filepaths
        slidenames_to_run = list(set([slideNameFromFilepath(fp) for fp in filepaths_to_run]))
        slides_to_run = [s for s in slides_to_run if s.name in slidenames_to_run]
        logstring+=f' from {len(slides_to_run)} slide(s): '
        for s in slides_to_run :
            logstring+=f'{s.name}, '
        logger.info(logstring[:-2])
        all_slide_filepaths = [fp for fp in all_slide_filepaths if slideNameFromFilepath(fp) in [s.name for s in slides_to_run]]
    #alert the user if exposure time corrections will be made
    if a.skip_exposure_time_correction :
        logger.info('Corrections for differences in exposure time will NOT be made')
    else :
        msg = 'Corrections for differences in exposure time will be made'
        msg+=f' based on the correction factors in {a.exposure_time_offset_file}'
        logger.info(msg)
    #alert the user if masking will be applied
    if a.skip_masking :
        logger.info('Images will NOT be masked before stacking')
    else :
        logger.info('Images WILL be masked before stacking')
    #alert the user if the thresholds will be calculated in this run
    if (not a.skip_masking) and a.threshold_file_dir is None :
        logstring = f'Background thresholds will be calculated for {len(slides_to_run)}'
        if len(slides_to_run)>1 :
            logstring+=' different slides:'
        else :
            logstring+=' slide'
        logstring+=': '
        for s in slides_to_run :
            logstring+=f'{s.name}, '
        logger.info(logstring[:-2])
    elif a.skip_masking :
        logger.info('Background thresholds will not be calculated because masking will be skipped')
    elif a.threshold_file_dir is not None :
        logger.info(f'Background thresholds will be read from the files at {a.threshold_file_dir}')
    #tell the user whether the tissue edge HPFs will be included
    if a.allow_edge_HPFs :
        logger.info('HPFs on tissue edges will be included in the image stack.')
    else :
        logger.info('HPFs on tissue edges will be found for each slide and excluded from the image stack.')
    #return the lists of filepaths and slidenames
    if len(all_slide_filepaths)<1 or len(filepaths_to_run)<1 or len(slides_to_run)<1 :
        raise RuntimeError('ERROR: The requested options have resulted in no slides or files to run!')
    return all_slide_filepaths, filepaths_to_run, slides_to_run

#helper function to do a whole run in any of the possible modes
def doRun(args,workingdir_path,logger=None) :
    #get the list of filepaths to run and the names of their slides
    if args.mode=='batch_flatfield' :
        all_filepaths = None; filepaths_to_run = None
        slides_to_run = [FlatfieldSlideInfo(sn,args.rawfile_top_dir,args.root_dir) for sn in split_csv_to_list(args.slides)]
    else :
        all_filepaths, filepaths_to_run, slides_to_run = getFilepathsAndSlidesToRun(args,logger)
    if args.mode=='check_run' :
        return
    #start up a flatfield producer
    ff_producer = FlatfieldProducer(slides_to_run,filepaths_to_run,workingdir_path,args.skip_exposure_time_correction,args.skip_masking,logger)
    #in batch_flatfield mode, just make the flatfield from all the files that already exist
    if args.mode=='batch_flatfield' :
        ff_producer.makeFlatfieldFromSlideMeanImages(args.batchID)
        ff_producer.writeOutInfo(args.batchID)
        fffn = f'{CONST.FLATFIELD_FILE_NAME_STEM}_BatchID_{args.batchID:02d}{CONST.FILE_EXT}'
        ff_producer.makeBatchFlatfieldSummaryPDF(args.root_dir,args.batchID)
        if os.path.isfile(os.path.join(workingdir_path,fffn)) :
            shutil.move(os.path.join(workingdir_path,fffn),os.path.join(os.path.dirname(workingdir_path),fffn))
        return
    #see if the code is running in batch mode (i.e. minimal output in automatic locations) and figure out the working directory path if so
    batch_mode = args.mode in ('slide_mean_image','batch_flatfield')
    #write out the text file of all the raw file paths that will be run
    if not batch_mode :
        ff_producer.writeFileLog(FILEPATH_TEXT_FILE_NAME)
        if args.mode=='choose_image_files' :
            return
    #Read in the exposure time correction offsets from the given directory
    if not args.skip_exposure_time_correction :
        ff_producer.readInExposureTimeCorrectionOffsets(args.exposure_time_offset_file)
    #Figure out the background thresholds per layer by looking at the HPFs on the tissue edges
    if not args.skip_masking :
        if args.threshold_file_dir is not None :
            ff_producer.readInBackgroundThresholds(args.threshold_file_dir)
        else :
            ff_producer.findBackgroundThresholds(all_filepaths,args.n_threads)
    if args.mode in ['slide_mean_image','make_flatfield', 'apply_flatfield'] :
        #mask and stack images together to make the mean image
        ff_producer.stackImages(args.n_threads,args.selected_pixel_cut,args.n_masking_images_per_slide,args.allow_edge_HPFs)
        ff_producer.makeMeanImage()
        if args.mode=='make_flatfield' :
            #make the flatfield image
            ff_producer.makeFlatField()
        elif args.mode=='apply_flatfield' :
            #apply the flatfield to the image stack
            with cd(args.prior_run_dir) :
                prior_run_ff_filename = (glob.glob(f'*{CONST.FLATFIELD_FILE_NAME_STEM}*{CONST.FILE_EXT}'))[0]
            ff_producer.applyFlatField(os.path.join(args.prior_run_dir,prior_run_ff_filename))
        #save the plots, etc.
        ff_producer.writeOutInfo()

#################### MAIN SCRIPT ####################
def main(args=None) :
    if args is None :
        #define and get the command-line arguments
        parser = ArgumentParser()
        #general positional arguments
        parser.add_argument('mode', 
                    choices=['slide_mean_image','batch_flatfield','make_flatfield','apply_flatfield','calculate_thresholds','check_run','choose_image_files'],                  
                            help='Which operation to perform')
        #the name of the working directory (optional because it's automatic in batch mode)
        parser.add_argument('--workingdir', 
                            help='Name of working directory to save created files in (set automatically in slide_mean_image and batch_flatfield modes)')
        #the batchID (optional because it's only needed in batch_flatfield mode)
        parser.add_argument('--batchID', type=int, default=-1,
                            help='BatchID for the created flatfield file and directory')
        #add the exposure time correction arguments
        et_correction_group = parser.add_mutually_exclusive_group()
        et_correction_group.add_argument('--exposure_time_offset_file',
                                         help="""Path to the .csv file specifying layer-dependent exposure time correction offsets for the slides in question
                                        [use this argument to apply corrections for differences in image exposure time]""")
        et_correction_group.add_argument('--skip_exposure_time_correction', action='store_true',
                                         help='Add this flag to entirely skip correcting image flux for exposure time differences')
        #add the group for defining the slide(s) to run on
        slide_definition_group = parser.add_argument_group('slide definition',
                                                            'what slides should be used, and where to find their files')
        slide_definition_group.add_argument('--slides',     
                                             help="""Path to .csv file listing FlatfieldSlideInfo objects (to use slides from multiple raw/root file paths),
                                                     or a comma-separated list of slide names to use with the common raw/root file paths""")
        slide_definition_group.add_argument('--rawfile_top_dir',
                                             help=f'Path to directory containing [slidename] subdirectories with raw "{UNIV_CONST.RAW_EXT}" files for all slides')
        slide_definition_group.add_argument('--root_dir',
                                             help='Path to Clinical_Specimen directory for all slides')
        #mutually exclusive group for how to handle the thresholding
        thresholding_group = parser.add_mutually_exclusive_group()
        thresholding_group.add_argument('--threshold_file_dir',
                                        help="""Path to the directory holding background threshold files created in previous runs for the slides in question
                                        [use this argument to re-use previously-calculated background thresholds]""")
        thresholding_group.add_argument('--skip_masking',       action='store_true',
                                        help="""Add this flag to entirely skip masking out the background regions of the images as they get added
                                        [use this argument to completely skip the background thresholding and masking]""")
        #group for how to select a subset of the slides' files
        file_selection_group = parser.add_argument_group('file selection',
                                                         'how many images from the slide set should be used, how to choose them, and where to find them')
        file_selection_group.add_argument('--prior_run_dir',     
                                          help="""Path to the working directory of a previous run whose raw files you want to use again, or whose calculated
                                          flatfield you want to apply to a different, orthogonal, set of files in the same slides""")
        file_selection_group.add_argument('--max_images',      default=-1,       type=int,         
                                          help='Number of images to load from the inputted list of slides')
        file_selection_group.add_argument('--selection_mode',  default='random', choices=['random','first','last'],
                                          help='Select "first", "last", or "random" (default) n images (where n=max_images) from the slide group.')
        file_selection_group.add_argument('--allow_edge_HPFs', action='store_true',
                                          help="""Add this flag to allow HPFs on the tissue edges to be stacked (not allowed by default)""")
        #group for some run options
        run_option_group = parser.add_argument_group('run options','other options for this run')
        run_option_group.add_argument('--n_threads',                   default=10,                         type=int,         
                                      help='Number of threads/processes to run at once in parallelized portions of the code')
        run_option_group.add_argument('--n_masking_images_per_slide',  default=0,                          type=int,         
                                      help='How many example masking images to save for each slide (randomly chosen)')
        run_option_group.add_argument('--selected_pixel_cut',          default=DEFAULT_SELECTED_PIXEL_CUT, type=float,         
                                      help='Minimum fraction (0->1) of pixels that must be selected as signal for an image to be added to the stack')
        run_option_group.add_argument('--other_runs_to_exclude',       default='',                         type=split_csv_to_list,
                                      help='Comma-separated list of additional, previously-run, working directories whose filepaths should be excluded')
        args = parser.parse_args()
    #make the working directory and start up the logger
    if args.mode=='slide_mean_image' :
        workingdir_path = getSlideMeanImageWorkingDirPath(FlatfieldSlideInfo((split_csv_to_list(args.slides))[0],args.rawfile_top_dir,args.root_dir))
    elif args.mode=='batch_flatfield' :
        workingdir_path = getBatchFlatfieldWorkingDirPath(args.root_dir,args.batchID)
    else :
        workingdir_path = os.path.abspath(os.path.normpath(args.workingdir))
    if not os.path.isdir(workingdir_path) :
        os.mkdir(workingdir_path)
    with RunLogger(args.mode,workingdir_path) as logger :
        #make sure the command line arguments make sense
        checkArgs(args)
        #do the whole run
        doRun(args,workingdir_path,logger)

if __name__=='__main__' :
    main()
    
