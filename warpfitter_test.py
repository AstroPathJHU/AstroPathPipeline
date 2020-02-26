#imports
from warpfitter import WarpFitter
import os

#details of the sample and overlaps to use
init_dir     = os.getcwd()
samp         = 'M21_1'
rawfile_dir  = os.path.join(os.path.sep,'Volumes','G','raw',samp)
root1_dir    = os.path.join(os.path.sep,'Volumes','G','heshy')
metafile_dir = os.path.join(root1_dir,samp,'dbload')
warp_dir     = os.path.join(init_dir,'warpfitter_test')
overlaps     = (24,24)

#make a warpfitter
fitter = WarpFitter(samp,rawfile_dir,metafile_dir,warp_dir,overlaps)

#Load the raw files
fitter.loadRawFiles()

#Fit the model to the data
result = fitter.doFit(fix_fxfy=True,fix_p1p2=True,print_every=10,show_plots=True)
print(result)
