# A script to run a quick example of the warpfitter

#imports
from astropath_calibration.exposuretime.exposure_time_fit_group import ExposureTimeOffsetFitGroup
from astropath_calibration.utilities.img_file_io import LayerOffset
from astropath_calibration.utilities.tableio import readtable
from .testbase import assertAlmostEqual
import pathlib, shutil

#some constants
folder = pathlib.Path(__file__).parent
slideID = 'M21_1'
rawfile_top_dir = folder/'data'/'raw'
root_dir = folder/'data'
working_dir = folder/'exposuretimefit_test_for_jenkins'
working_dir.mkdir(exist_ok=True)

layers_arg = [26,33]
overlaps_arg = [46]
smoothsigma_arg=3
initial_offset_arg = 5
min_pixel_frac_arg = 1e-4
max_iter_arg = 15000
gtol_arg=1e-8
eps_arg=0.25
print_every_arg = 10
n_comparisons_to_save_arg = 1

#initialize a fit
print('Defining group of fits....')
fit_group = ExposureTimeOffsetFitGroup(slideID,rawfile_top_dir,root_dir,working_dir,layers_arg,1)
#run the fits
print('Running fits....')
fit_group.runFits(None,overlaps_arg,smoothsigma_arg,False,
                  initial_offset_arg,min_pixel_frac_arg,max_iter_arg,gtol_arg,eps_arg,print_every_arg,
                  n_comparisons_to_save_arg,True)
new = readtable(working_dir/f"{slideID}_layers_{layers_arg[0]}-{layers_arg[-1]}_best_fit_offsets_exposuretimefit_test_for_jenkins.csv", LayerOffset)
ref = readtable(folder/"reference"/"exposuretimefit"/f"{slideID}_layers_{layers_arg[0]}-{layers_arg[-1]}_best_fit_offsets_exposuretimefit_test_for_jenkins.csv", LayerOffset)
for offsetnew, offsetref in zip(new, ref):
  assertAlmostEqual(offsetnew, offsetref, rtol=1e-4, atol=1e-8)

print('Removing working directory...')
shutil.rmtree(working_dir,ignore_errors=True)
print('Done!')
