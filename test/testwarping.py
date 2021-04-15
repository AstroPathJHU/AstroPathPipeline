# a small script to test the results of some warps applied to a few images

#imports
from astropath.hpfs.warping.warp import PolyFieldWarp, CameraWarp
import pathlib, numpy as np

#constants

#slide name and directories
slide_ID = r"M21_1"
folder = (pathlib.Path(__file__)).parent

#file stems
file1stem = slide_ID+'_[46163,12453]'
file2stem = slide_ID+'_[46698,12453]'

# .raw file paths
rawfile1path = folder / 'data' / 'raw' / slide_ID / f'{file1stem}.Data.dat'
rawfile2path = folder / 'data' / 'raw' / slide_ID / f'{file2stem}.Data.dat'

# warp test file paths
warp1file1layer1p=folder / 'reference' / 'warping' / f'{file1stem}.fieldWarp_layer01'
warp1file1layer2p=folder / 'reference' / 'warping' / f'{file1stem}.fieldWarp_layer02'
warp1file2layer1p=folder / 'reference' / 'warping' / f'{file2stem}.fieldWarp_layer01'
warp1file2layer2p=folder / 'reference' / 'warping' / f'{file2stem}.fieldWarp_layer02'
warp2file1layer1p=folder / 'reference' / 'warping' / f'{file1stem}.camWarp_layer01'
warp2file1layer2p=folder / 'reference' / 'warping' / f'{file1stem}.camWarp_layer02'
warp2file2layer1p=folder / 'reference' / 'warping' / f'{file2stem}.camWarp_layer01'
warp2file2layer2p=folder / 'reference' / 'warping' / f'{file2stem}.camWarp_layer02'

#first make a few warps
warp1 = PolyFieldWarp() #Alex's default polynomial field warp
warp2 = CameraWarp(cx=584.,cy=600.,k1=15.,k2=-6800.,p1=0.002,p2=-0.002) #Modified camera warp

#get the raw image layers
file1_raw = warp1.getHWLFromRaw(rawfile1path)
file2_raw = warp1.getHWLFromRaw(rawfile2path)

#read in the example files
warp1file1layer1ex = warp1.getSingleLayerImage(warp1file1layer1p)
warp1file1layer2ex = warp1.getSingleLayerImage(warp1file1layer2p)
warp1file2layer1ex = warp1.getSingleLayerImage(warp1file2layer1p)
warp1file2layer2ex = warp1.getSingleLayerImage(warp1file2layer2p)
warp2file1layer1ex = warp2.getSingleLayerImage(warp2file1layer1p)
warp2file1layer2ex = warp2.getSingleLayerImage(warp2file1layer2p)
warp2file2layer1ex = warp2.getSingleLayerImage(warp2file2layer1p)
warp2file2layer2ex = warp2.getSingleLayerImage(warp2file2layer2p)

#warp two layers each from the raw files
#with warp 1
warp1file1layer1 = warp1.getWarpedLayer(file1_raw[:,:,0])
warp1file1layer2 = warp1.getWarpedLayer(file1_raw[:,:,1])
warp1file2layer1 = warp1.getWarpedLayer(file2_raw[:,:,0])
warp1file2layer2 = warp1.getWarpedLayer(file2_raw[:,:,1])
#with warp 2
warp2file1layer1 = warp2.getWarpedLayer(file1_raw[:,:,0])
warp2file1layer2 = warp2.getWarpedLayer(file1_raw[:,:,1])
warp2file2layer1 = warp2.getWarpedLayer(file2_raw[:,:,0])
warp2file2layer2 = warp2.getWarpedLayer(file2_raw[:,:,1])

#make sure the warped images are identical to the examples
np.testing.assert_array_equal(warp1file1layer1,warp1file1layer1ex)
np.testing.assert_array_equal(warp1file1layer2,warp1file1layer2ex)
np.testing.assert_array_equal(warp1file2layer1,warp1file2layer1ex)
np.testing.assert_array_equal(warp1file2layer2,warp1file2layer2ex)
np.testing.assert_array_equal(warp2file1layer1,warp2file1layer1ex)
np.testing.assert_array_equal(warp2file1layer2,warp2file1layer2ex)
np.testing.assert_array_equal(warp2file2layer1,warp2file2layer1ex)
np.testing.assert_array_equal(warp2file2layer2,warp2file2layer2ex)
print('All tests passed!')
