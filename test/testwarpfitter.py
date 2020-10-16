# A script to run a quick example of the warpfitter

#imports
from ..utilities.tableio import readtable
from ..warping.utilities import WarpFitResult
from ..warping.warp_fitter import WarpFitter
from .testbase import assertAlmostEqual
import pathlib, shutil

#some constants
folder = pathlib.Path(__file__).parent
samp = 'M21_1'
rawfile_top_dir = folder/'data'/'raw'
metadata_top_dir = folder/'data'
working_dir = folder/'warpfitter_test_for_jenkins'
working_dir.mkdir(exist_ok=True)
overlaps = [46]
layer = 1

fixed_arg = ['fx','fy','p1','p2']
normalize_arg = ['cx','cy','fx','fy','k1','k2','k3','p1','p2']
init_pars_arg = {'cx':596.5,'cy':601.2,'k1':10.25,'k2':8913.,'k3':-10980000.}
init_bounds_arg = {'cx':(529.3,663.7),'cy':(551.0,651.4)}
max_radial_warp = 8.
max_tangential_warp = 4.
p1p2_polish_lasso_lambda = 1.0
print_every = 25
max_iter = 1

#make the WarpFitter Objects
print('Initializing WarpFitter')
fitter = WarpFitter(samp,rawfile_top_dir,metadata_top_dir,working_dir,overlaps,layer)
#load the raw files
print('Loading raw files')
fitter.loadRawFiles(None,None,1)
#fit the model to the data
print('Running doFit')
result = fitter.doFit(fixed=fixed_arg,normalize=normalize_arg,init_pars=init_pars_arg,init_bounds=init_bounds_arg,float_p1p2_in_polish_fit=True,
                      max_radial_warp=max_radial_warp,max_tangential_warp=max_tangential_warp,p1p2_polish_lasso_lambda=p1p2_polish_lasso_lambda,
                      polish=True,print_every=print_every,maxiter=max_iter,save_fields=False)
new = readtable(working_dir/"fit_result.csv", WarpFitResult)
ref = readtable(folder/"reference"/"warping"/"fit_result.csv", WarpFitResult)
for resultnew, resultref in zip(new, ref):
  resultnew.dirname = resultref.dirname = ""
  resultnew.global_fit_its = resultref.global_fit_its = \
  resultnew.global_fit_time = resultref.global_fit_time = \
  resultnew.polish_fit_its = resultref.polish_fit_its = \
  resultnew.polish_fit_time = resultref.polish_fit_time = \
  0
  assertAlmostEqual(resultnew, resultref, rtol=1e-3)

print('Removing working directory...')
shutil.rmtree(working_dir,ignore_errors=True)
print('Done!')
