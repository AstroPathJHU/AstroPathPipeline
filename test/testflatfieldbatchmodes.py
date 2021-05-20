# A script to run tests of the flatfield batch modes

#imports
from astropath.hpfs.flatfield.run_flatfield import main
from astropath.hpfs.flatfield.utilities import flatfield_logger, FlatfieldSlideInfo, getSlideMeanImageWorkingDirPath, getBatchFlatfieldWorkingDirPath
from astropath.utilities.misc import cd
from argparse import Namespace
import pathlib, shutil, subprocess

#some constants
folder = pathlib.Path(__file__).parent
slide_ID = 'M21_1'

#Run the slide_mean_image test
flatfield_logger.info('TESTING slide_mean_image')
args = Namespace(
        mode='slide_mean_image',
        filetype='raw',
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
main(args)
#check the logfile for error messages
slide = FlatfieldSlideInfo(slide_ID,str(folder/'data'/'raw'),str(folder/'data'))
slide_mean_image_working_dir = getSlideMeanImageWorkingDirPath(slide,'raw')
with open(pathlib.Path(f'{slide_mean_image_working_dir}/global-slide_mean_image.log'),'r') as fp :
    for l in fp.readlines() :
        if 'ERROR' in l :
            raise RuntimeError('ERROR: there were errors during the slide_mean_image test; check the log files for what went wrong!')

#Run the batch_flatfield test
flatfield_logger.info('TESTING batch_flatfield')
ff_dirpath = folder/'data'/'flatfield'
if not pathlib.Path.is_dir(ff_dirpath) :
    pathlib.Path.mkdir(ff_dirpath)
args = Namespace(
        mode='batch_flatfield',
        filetype='raw',
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
main(args)
#check the logfile for error messages
batch_flatfield_working_dir = getBatchFlatfieldWorkingDirPath(folder/'data',1)
with open(pathlib.Path(f'{batch_flatfield_working_dir}/global-batch_flatfield.log'),'r') as fp :
    for l in fp.readlines() :
        if 'ERROR' in l :
            raise RuntimeError('ERROR: there were errors during the batch_flatfield test; check the log files for what went wrong!')

#remove the working directory and the logs that were created
flatfield_logger.info('Removing working directories and logfiles....')
with cd(slide_mean_image_working_dir):
  subprocess.run(["git", "clean", "-fd"])
  subprocess.run(["git", "checkout", "--", "."])
(folder/'data'/'logfiles'/'slide_mean_image.log').unlink()
(folder/'data'/f'{slide_ID}'/'logfiles'/f'{slide_ID}-slide_mean_image.log').unlink()
shutil.rmtree(batch_flatfield_working_dir,ignore_errors=True)
(folder/'data'/'logfiles'/'batch_flatfield.log').unlink()
(folder/'data'/f'{slide_ID}'/'logfiles'/f'{slide_ID}-batch_flatfield.log').unlink()
shutil.rmtree(ff_dirpath)
flatfield_logger.info('All Done!')
