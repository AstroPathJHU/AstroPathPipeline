# A script to run a quick example of the warpfitter

#imports
from ..exposuretime.exposure_time_fit_group import ExposureTimeOffsetFitGroup
import pathlib, shutil

#some constants
folder = pathlib.Path(__file__).parent
samp = 'M21_1'
rawfile_top_dir = folder/'data'/'raw'
metadata_top_dir = folder/'data'
working_dir = folder/'exposuretimefit_test_for_jenkins'
working_dir.mkdir(exist_ok=True)
overlaps_arg = [46]
layers_arg = [26,33]

#################################################
#copy/paste the run_exposure_time_fits code here#
#################################################
print('Removing working directory...')
shutil.rmtree(working_dir,ignore_errors=True)
print('Done!')