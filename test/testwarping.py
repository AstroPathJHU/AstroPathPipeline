# a small script to test the results of some warps applied to a few images

#imports
from ..warp import PolyFieldWarp, CameraWarp
import os, numpy as np

#constants

#sample name and directories
samp = r"M21_1"
folder = os.path.dirname(__file__)
root1_dir = os.path.join(folder,'data')
root2_dir = os.path.join(folder,'data','flatw')
rawfile_dir = os.path.join(folder,'data','raw')

#file stems
file1stem = samp+'_[46163,12453]'
file2stem = samp+'_[46698,12453]'

# .raw file paths
rawfile1path = os.path.join(folder,'data','raw',samp,file1stem+'.raw')
rawfile2path = os.path.join(folder,'data','raw',samp,file2stem+'.raw')

# warp test file paths
warp1file1layer1p=os.path.join(folder,'warpingreference',file1stem+'.fieldWarp_layer01')
warp1file1layer2p=os.path.join(folder,'warpingreference',file1stem+'.fieldWarp_layer02')
warp1file2layer1p=os.path.join(folder,'warpingreference',file2stem+'.fieldWarp_layer01')
warp1file2layer2p=os.path.join(folder,'warpingreference',file2stem+'.fieldWarp_layer02')
warp2file1layer1p=os.path.join(folder,'warpingreference',file1stem+'.camWarp_layer01')
warp2file1layer2p=os.path.join(folder,'warpingreference',file1stem+'.camWarp_layer02')
warp2file2layer1p=os.path.join(folder,'warpingreference',file2stem+'.camWarp_layer01')
warp2file2layer2p=os.path.join(folder,'warpingreference',file2stem+'.camWarp_layer02')

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
