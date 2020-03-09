# A script to run a quick example of the warpfitter

#imports
from ..warpfitter import WarpFitter
import os

#some constants
folder = os.path.dirname(__file__)
samp = 'M21_1'
rawfile_dir = os.path.join(folder,'data','raw')
metafile_dir = os.path.join(folder,'data',samp,'dbload')
working_dir = os.path.join('test','warpfitter_test_for_jenkins')
overlaps = [46]
layers = [1]
fix_cxcy = False
fix_fxfy = True
fix_k1k2 = False
fix_p1p2 = False
max_radial_warp = 15.
max_tangential_warp = 15.
print_every = 10
max_iter = 1

#make the WarpFitter Objects
print('Initializing WarpFitter')
fitter = WarpFitter(samp,rawfile_dir,metafile_dir,working_dir,overlaps,layers)
#load the raw files
print('Loading raw files')
fitter.loadRawFiles()
#fit the model to the data
print('Running doFit')
result = fitter.doFit(fix_cxcy=fix_cxcy,fix_fxfy=fix_fxfy,fix_k1k2=fix_k1k2,fix_p1p2=fix_p1p2,
                      max_radial_warp=max_radial_warp,max_tangential_warp=max_tangential_warp,
                      print_every=print_every,maxiter=max_iter)
print(f'result:\n{result}')
os.system(f'cat {os.path.join(working_dir,"warping_parameters.txt")}; rm -rf {working_dir} ')
print('Done!')