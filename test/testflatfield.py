# A script to do a test run of a flatfield

#imports
from astropath.hpfs.flatfield.run_flatfield import main
from astropath.hpfs.flatfield.utilities import flatfield_logger
from argparse import Namespace
import pathlib, shutil

#some constants
folder = pathlib.Path(__file__).parent
slide_ID = 'M21_1'

#First make a flatfield
flatfield_logger.info('TESTING make_flatfield')
make_flatfield_workingdir_name = 'make_flatfield_test_for_jenkins'
make_flatfield_working_dir = folder/make_flatfield_workingdir_name
make_flatfield_working_dir.mkdir(exist_ok=True)
args = Namespace(
        mode='make_flatfield',
        filetype='raw',
        exposure_time_offset_file=str(folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'),
        skip_exposure_time_correction=False,
        workingdir=str(make_flatfield_working_dir),
        batchID=-1,
        slides=f'{slide_ID}',
        rawfile_top_dir=str(folder/'data'/'raw'),
        root_dir=str(folder/'data'),
        threshold_file_dir=None,#str(folder/'data'/'corrections'),
        skip_masking=False,
        prior_run_dir=None,
        max_images=-1,
        selection_mode='random',
        allow_edge_HPFs=True,
        n_threads=10,
        n_masking_images_per_slide=0,
        selected_pixel_cut=0.0,
        other_runs_to_exclude=['']
    )
main(args)
#overwrite the file log so that applying the flatfield doesn't crash because it's using the same files
flp = make_flatfield_working_dir/'filepath_log.txt'
flp.unlink()
with open(flp,'w') as fp :
    fp.write('hello : )\n')

#Then apply it to the same images
flatfield_logger.info('TESTING apply_flatfield')
apply_flatfield_workingdir_name = 'apply_flatfield_test_for_jenkins'
apply_flatfield_working_dir = folder/apply_flatfield_workingdir_name
apply_flatfield_working_dir.mkdir(exist_ok=True)
args = Namespace(
        mode='apply_flatfield',
        filetype='raw',
        exposure_time_offset_file=str(folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'),
        skip_exposure_time_correction=False,
        workingdir=str(apply_flatfield_working_dir),
        batchID=-1,
        slides=f'{slide_ID}',
        rawfile_top_dir=str(folder/'data'/'raw'),
        root_dir=str(folder/'data'),
        threshold_file_dir=str(folder/'data'/'corrections'),
        skip_masking=False,
        prior_run_dir=str(make_flatfield_working_dir),
        max_images=-1,
        selection_mode='random',
        allow_edge_HPFs=True,
        n_threads=10,
        n_masking_images_per_slide=0,
        selected_pixel_cut=0.0,
        other_runs_to_exclude=['']
    )
main(args)

#get rid of both working directories
flatfield_logger.info('Removing working directories....')
shutil.rmtree(make_flatfield_working_dir,ignore_errors=True)
shutil.rmtree(apply_flatfield_working_dir,ignore_errors=True)
flatfield_logger.info('All Done!')
