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

layers_arg = [26,33]
overlaps_arg = [46]
smoothsigma_arg=3
initial_offset_arg = 25
bounds_arg = (0,1000)
max_iter_arg = 10
gtol_arg=1e-5
eps_arg=2
print_every_arg = 2
n_comparisons_to_save_arg = 1

#initialize a fit
print('Defining group of fits....')
fit_group = ExposureTimeOffsetFitGroup(samp,rawfile_top_dir,metadata_top_dir,working_dir,layers_arg,1)
#prepare the fits to run by reading the files and exposure times etc.
print('Loading files and preparing fits....')
fit_group.prepFits(None,overlaps_arg,smooth_sigma_arg,True)
#run the fits
print('Running fits....')
fit_group.runFits(initial_offset_arg,bounds_arg,max_iter_arg,gtol_arg,eps_arg,print_every_arg)
#write out the fit results
print('Writing out fit results....')
fit_group.writeOutResults(n_comparisons_to_save_arg)
print('Removing working directory...')
shutil.rmtree(working_dir,ignore_errors=True)
print('Done!')