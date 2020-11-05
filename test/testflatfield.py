# A script to run a quick test run of a flatfield

#imports
from ..flatfield.run_flatfield import checkArgs, doRun
from ..flatfield.logging import RunLogger
from ..flatfield.utilities import flatfield_logger, FlatfieldSlideInfo, getSlideMeanImageWorkingDirPath, getBatchFlatfieldWorkingDirPath
from argparse import Namespace
import pathlib, os, shutil

#some constants
folder = pathlib.Path(__file__).parent
slide_ID = 'M21_1'
rawfile_top_dir = folder/'data'/'raw'
rawfile_ext = '.Data.dat'
root_dir = folder/'data'
dims = (1004,1344,35)

##First make a flatfield
#flatfield_logger.info('TESTING make_flatfield')
#make_flatfield_workingdir_name = 'make_flatfield_test_for_jenkins'
#make_flatfield_working_dir = folder/make_flatfield_workingdir_name
#make_flatfield_working_dir.mkdir(exist_ok=True)
#args = Namespace(
#        mode='make_flatfield',
#        exposure_time_offset_file=str(folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'),
#        skip_exposure_time_correction=False,
#        workingdir=str(make_flatfield_working_dir),
#        batchID=None,
#        slides=f'{slide_ID}',
#        rawfile_top_dir=str(folder/'data'/'raw'),
#        root_dir=str(folder/'data'),
#        threshold_file_dir=None,
#        skip_masking=True,
#        prior_run_dir=None,
#        max_images=-1,
#        selection_mode='random',
#        allow_edge_HPFs=True,
#        n_threads=10,
#        n_masking_images_per_slide=0,
#        selected_pixel_cut=0.8,
#        other_runs_to_exclude=['']
#    )
#checkArgs(args)
#doRun(args,make_flatfield_working_dir,None)
##overwrite the file log so that applying the flatfield doesn't crash because it's using the same files
#flp = make_flatfield_working_dir/'filepath_log.txt'
#os.remove(flp)
#with open(flp,'w') as fp :
#    fp.write('hello : )\n')
#
##Then apply it to the same images
#flatfield_logger.info('TESTING apply_flatfield')
#apply_flatfield_workingdir_name = 'apply_flatfield_test_for_jenkins'
#apply_flatfield_working_dir = folder/apply_flatfield_workingdir_name
#apply_flatfield_working_dir.mkdir(exist_ok=True)
#args = Namespace(
#        mode='apply_flatfield',
#        exposure_time_offset_file=str(folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'),
#        skip_exposure_time_correction=False,
#        workingdir=str(apply_flatfield_working_dir),
#        batchID=None,
#        slides=f'{slide_ID}',
#        rawfile_top_dir=str(folder/'data'/'raw'),
#        root_dir=str(folder/'data'),
#        threshold_file_dir=None,
#        skip_masking=True,
#        prior_run_dir=str(make_flatfield_working_dir),
#        max_images=-1,
#        selection_mode='random',
#        allow_edge_HPFs=True,
#        n_threads=10,
#        n_masking_images_per_slide=0,
#        selected_pixel_cut=0.8,
#        other_runs_to_exclude=['']
#    )
#checkArgs(args)
#doRun(args,apply_flatfield_working_dir,None)
#
##get rid of both working directories
#flatfield_logger.info('Removing working directories....')
#shutil.rmtree(make_flatfield_working_dir,ignore_errors=True)
#shutil.rmtree(apply_flatfield_working_dir,ignore_errors=True)

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
