# A script to run a quick example of the warpfitter

#imports
from astropath.hpfs.warping.run_warp_fitter import main
from astropath.hpfs.warping.utilities import WarpFitResult
from astropath.utilities.tableio import readtable
from .testbase import assertAlmostEqual
from argparse import Namespace
import pathlib, shutil

#some constants
folder = pathlib.Path(__file__).parent
working_dir = folder/'warpfitter_test_for_jenkins'

args = Namespace(
        mode='warp_fit',
        slideID='M21_1',
        rawfile_top_dir=str(folder/'data'/'raw'),
        root_dir=str(folder/'data'),
        workingdir=str(working_dir),
        exposure_time_offset_file=str(folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'),
        skip_exposure_time_correction=False,
        flatfield_file=None,
        skip_flatfielding=True,
        max_iter=100,
        normalize='cx,cy,fx,fy,k1,k2,k3,p1,p2',
        max_radial_warp=8.,
        max_tangential_warp=4.,
        print_every=25,
        fixed='fx,fy,p1,p2',
        init_pars='cx=596.5,cy=601.2,k1=10.25,k2=8913.,k3=-10980000.',
        init_bounds='cx=529.3:663.7,cy=551.0:651.4',
        float_p1p2_to_polish=True,
        p1p2_polish_lasso_lambda=1.0,
        octet_file=None,
        octet_run_dir=None,
        threshold_file_dir=None,
        req_pixel_frac=0.85,
        octet_selection='random_2',
        workers=None,
        layer=1,
        overlaps=[46],
        octets=[-999],
        save_warp_fields=False
    )
main(args)

new = readtable(working_dir/"fit_result.csv", WarpFitResult, checkorder=True, checknewlines=True)
ref = readtable(folder/"reference"/"warping"/"fit_result.csv", WarpFitResult, checkorder=True, checknewlines=True)
ref2 = readtable(folder/"reference"/"warping"/"fit_result_2.csv", WarpFitResult, checkorder=True, checknewlines=True)
for resultnew, resultref, resultref2 in zip(new, ref, ref2):
  for result in resultnew, resultref, resultref2:
    result.dirname = ""
    result.global_fit_its = result.global_fit_time = \
    result.polish_fit_its = result.polish_fit_time = \
    0
  try:
    assertAlmostEqual(resultnew, resultref, rtol=5e-3)
  except AssertionError:
    assertAlmostEqual(resultnew, resultref2, rtol=5e-3)

print('Removing working directory...')
shutil.rmtree(working_dir,ignore_errors=True)
print('Done!')
