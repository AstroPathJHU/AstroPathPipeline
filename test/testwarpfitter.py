# A script to run a quick example of the warpfitter

#imports
from ..warping.warpfitter import WarpFitter
import pathlib, shutil

#some constants
folder = pathlib.Path(__file__).parent
samp = 'M21_1'
rawfile_top_dir = folder/'data'/'raw'
dbload_top_dir = folder/'data'
working_dir = folder/'warpfitter_test_for_jenkins'
working_dir.mkdir(exist_ok=True)
overlaps = [46]
layer = 1

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
result = fitter.doFit(fixed=['fx','fy','p1','p2'],normalize=['cx','cy','fx','fy','k1','k2','k3','p1','p2'],float_p1p2_in_polish_fit=True,
					  max_radial_warp=max_radial_warp,max_tangential_warp=max_tangential_warp,p1p2_polish_lasso_lambda=p1p2_polish_lasso_lambda,
                      polish=True,print_every=print_every,maxiter=max_iter)
print('Removing working directory...')
shutil.rmtree(working_dir,ignore_errors=True)
print('Done!')
