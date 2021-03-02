# A script to run exposure time fit tests

#imports
from astropath_calibration.exposuretime.run_exposure_time_fits import main
from astropath_calibration.utilities.img_file_io import LayerOffset
from astropath_calibration.utilities.tableio import readtable
from argparse import Namespace
from .testbase import assertAlmostEqual
import pathlib, shutil

#some constants
folder = pathlib.Path(__file__).parent
working_dir = folder/'exposuretimefit_test_for_jenkins'

#arguments namespace
args = Namespace(
        slideID='M21_1',
        rawfile_top_dir=str(folder/'data'/'raw'),
        root_dir=str(folder/'data'),
        workingdir=str(working_dir),
        flatfield_file=None,
        skip_flatfielding=True,
        smooth_sigma=3.,
        use_whole_image=False,
        initial_offset=5.,
        min_pixel_frac_for_offset_limit=1e-4,
        max_iter=15000,
        gtol=1e-8,
        eps=0.25,
        print_every=10,
        n_threads=5,
        layers=[26,33],
        overlaps=[46],
        n_comparisons_to_save=1,
        allow_edge_HPFs=True,
    )

#run the main function
main(args)

#check the results against the reference
print('Checking fit results....')
new = readtable(args.working_dir/f"{args.slideID}_layers_{args.layers[0]}-{args.layers[-1]}_best_fit_offsets_exposuretimefit_test_for_jenkins.csv", LayerOffset)
ref = readtable(folder/"reference"/"exposuretimefit"/f"{args.slideID}_layers_{args.layers[0]}-{args.layers[-1]}_best_fit_offsets_exposuretimefit_test_for_jenkins.csv", LayerOffset)
for offsetnew, offsetref in zip(new, ref):
  assertAlmostEqual(offsetnew, offsetref, rtol=1e-4, atol=1e-8)

print('Removing working directory...')
shutil.rmtree(working_dir,ignore_errors=True)
print('Done!')
