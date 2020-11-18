# A script to run a quick test run of a flatfield

#imports
from astropath_calibration.flatfield.run_flatfield import checkArgs, doRun
from astropath_calibration.flatfield.logging import RunLogger
from astropath_calibration.flatfield.utilities import flatfield_logger, FlatfieldSlideInfo, getSlideMeanImageWorkingDirPath, getBatchFlatfieldWorkingDirPath
from argparse import Namespace
import pathlib, os, shutil

#some constants
folder = pathlib.Path(__file__).parent
slide_ID = 'M21_1'
rawfile_top_dir = folder/'data'/'raw'
rawfile_ext = '.Data.dat'
root_dir = folder/'data'
dims = (1004,1344,35)

#Run the slide_mean_image test
flatfield_logger.info('TESTING slide_mean_image')
slide = FlatfieldSlideInfo(slide_ID,str(folder/'data'/'raw'),str(folder/'data'))
slide_mean_image_working_dir = getSlideMeanImageWorkingDirPath(slide)
args = Namespace(
        mode='slide_mean_image',
        exposure_time_offset_file=str(folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'),
        skip_exposure_time_correction=False,
        workingdir=None,
        batchID=-1,
        slides=f'{slide_ID}',
        rawfile_top_dir=str(folder/'data'/'raw'),
        root_dir=str(folder/'data'),
        threshold_file_dir=None,
        skip_masking=False,
        prior_run_dir=None,
        max_images=-1,
        selection_mode='random',
        allow_edge_HPFs=False,
        n_threads=10,
        n_masking_images_per_slide=0,
        selected_pixel_cut=0.8,
        other_runs_to_exclude=['']
    )
with RunLogger(args.mode,slide_mean_image_working_dir) as logger :
    checkArgs(args)
    doRun(args,slide_mean_image_working_dir,logger)
#check the logfile for error messages
with open(os.path.join(slide_mean_image_working_dir,'global-slide_mean_image.log'),'r') as fp :
    for l in fp.readlines() :
        if 'ERROR' in l :
            raise RuntimeError('ERROR: there were errors during the slide_mean_image test; check the log files for what went wrong!')

#Run the batch_flatfield test
flatfield_logger.info('TESTING batch_flatfield')
ff_dirpath = folder/'data'/'Flatfield'
if not os.path.isdir(ff_dirpath) :
    os.mkdir(ff_dirpath)
batch_flatfield_working_dir = getBatchFlatfieldWorkingDirPath(folder/'data',1)
args = Namespace(
        mode='batch_flatfield',
        exposure_time_offset_file=None,
        skip_exposure_time_correction=False,
        workingdir=None,
        batchID=1,
        slides=f'{slide_ID}',
        rawfile_top_dir=str(folder/'data'/'raw'),
        root_dir=str(folder/'data'),
        threshold_file_dir=None,
        skip_masking=False,
        prior_run_dir=None,
        max_images=-1,
        selection_mode='random',
        allow_edge_HPFs=False,
        n_threads=10,
        n_masking_images_per_slide=0,
        selected_pixel_cut=0.8,
        other_runs_to_exclude=['']
    )
with RunLogger(args.mode,batch_flatfield_working_dir) as logger :
    checkArgs(args)
    doRun(args,batch_flatfield_working_dir,logger)
#check the logfile for error messages
with open(os.path.join(batch_flatfield_working_dir,'global-batch_flatfield.log'),'r') as fp :
    for l in fp.readlines() :
        if 'ERROR' in l :
            raise RuntimeError('ERROR: there were errors during the batch_flatfield test; check the log files for what went wrong!')

#remove the working directory and the logs that were created
flatfield_logger.info('Removing working directories and logfiles....')
shutil.rmtree(slide_mean_image_working_dir,ignore_errors=True)
os.remove(folder/'data'/'logfiles'/'slide_mean_image.log')
os.remove(folder/'data'/f'{slide_ID}'/'logfiles'/f'{slide_ID}-slide_mean_image.log')
shutil.rmtree(batch_flatfield_working_dir,ignore_errors=True)
os.remove(folder/'data'/'logfiles'/'batch_flatfield.log')
os.remove(folder/'data'/f'{slide_ID}'/'logfiles'/f'{slide_ID}-batch_flatfield.log')
shutil.rmtree(ff_dirpath)
flatfield_logger.info('All Done!')