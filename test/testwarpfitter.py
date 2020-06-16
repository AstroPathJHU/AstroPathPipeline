# A script to run a quick example of the warpfitter

#imports
from ..warping.warpfitter import WarpFitter
import pathlib

#some constants
folder = pathlib.Path(__file__).parent
samp = 'M21_1'
rawfile_top_dir = folder/'data'/'raw'
dbload_top_dir = folder/'data'
working_dir = folder/'warpfitter_test_for_jenkins'
working_dir.mkdir(exist_ok=True)
overlaps = [46]
layer = 1
fix_cxcy   = False
fix_fxfy   = True
fix_k1k2k3 = False
fix_p1p2_in_global_fit = True
fix_p1p2_in_polish_fit = False

max_radial_warp = 8.
max_tangential_warp = 4.
p1p2_polish_lasso_lambda = 0.001
print_every = 1
max_iter = 1

#make the WarpFitter Objects
print('Initializing WarpFitter')
fitter = WarpFitter(samp,rawfile_top_dir,dbload_top_dir,working_dir,overlaps,layer)
#load the raw files
print('Loading raw files')
fitter.loadRawFiles()
#fit the model to the data
print('Running doFit')
result = fitter.doFit(fix_cxcy=fix_cxcy,fix_fxfy=fix_fxfy,fix_k1k2k3=fix_k1k2k3,
					  fix_p1p2_in_global_fit=fix_p1p2_in_global_fit,fix_p1p2_in_polish_fit=fix_p1p2_in_polish_fit,
                      max_radial_warp=max_radial_warp,max_tangential_warp=max_tangential_warp,
                      polish=True,print_every=print_every,maxiter=max_iter)
print(f'result:\n{result}')
with open(working_dir/"warping_parameters.txt") as f:
  print(f.read())
print('Done!')
