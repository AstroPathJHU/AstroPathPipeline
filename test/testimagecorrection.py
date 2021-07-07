#imports
from astropath.hpfs.image_correction.run_image_correction import main
from astropath.utilities.img_file_io import get_raw_as_hwl, get_raw_as_hw
from astropath.utilities.misc import cd
import numpy as np
from argparse import Namespace
import pathlib, glob, shutil

#some constants
folder = pathlib.Path(__file__).parent
workingdir_name = 'image_correction'
working_dir = folder/'test_for_jenkins'/workingdir_name
working_dir.mkdir(exist_ok=True, parents=True)
multilayer_ref_path = folder/'data'/'reference'/'imagecorrection'/'multilayer'
singlelayer_ref_path = folder/'data'/'reference'/'imagecorrection'/'singlelayer'
dims = (1004,1344,35)


#start by running for all layers at once
print('TESTING MULTILAYER CORRECT/COPY')
args = Namespace(
        slideID='M21_1',
        rawfile_top_dir=str(folder/'data'/'raw'),
        root_dir=str(folder/'data'),
        workingdir=str(working_dir),
        flatfield_file=None,
        skip_flatfielding=True,
        warp_def=str(folder/'data'/'corrections'/'TEST_WARPING_weighted_average_warp.csv'),
        skip_warping=False,
        warp_shift_file=str(folder/'data'/'corrections'/'random_warp_shifts_for_testing.csv'),
        warp_shift=None,
        warping_scalefactor=1.0,
        layer=-1,
        input_file_extension='.Data.dat',
        output_file_extension='.fw',
        max_files=-1,
    )
main(args)
#check the output image files
with cd(working_dir) :
    img_filenames = glob.glob('*.fw')
for imfn in img_filenames :
    test_img = get_raw_as_hwl(working_dir/imfn,dims[0],dims[1],dims[2])
    ref_img = get_raw_as_hwl(multilayer_ref_path/imfn,dims[0],dims[1],dims[2])
    np.testing.assert_array_equal(test_img,ref_img)
#remove the working directory
shutil.rmtree(working_dir,ignore_errors=True)

#run again with only the first layer
print('TESTING SINGLE LAYER CORRECT/COPY')
working_dir.mkdir(exist_ok=True)
args = Namespace(
        slideID='M21_1',
        rawfile_top_dir=str(folder/'data'/'raw'),
        root_dir=str(folder/'data'),
        workingdir=str(working_dir),
        flatfield_file=None,
        skip_flatfielding=True,
        warp_def=str(folder/'data'/'corrections'/'TEST_WARPING_weighted_average_warp.csv'),
        skip_warping=False,
        warp_shift_file=str(folder/'data'/'corrections'/'random_warp_shifts_for_testing.csv'),
        warp_shift=None,
        warping_scalefactor=1.0,
        layer=1,
        input_file_extension='.Data.dat',
        output_file_extension='.fw',
        max_files=-1,
    )
main(args)
#check the output image files
with cd(working_dir) :
    img_filenames = glob.glob('*.fw01')
for imfn in img_filenames :
    test_img = get_raw_as_hw(working_dir/imfn,dims[0],dims[1])
    ref_img = get_raw_as_hw(singlelayer_ref_path/imfn,dims[0],dims[1])
    np.testing.assert_array_equal(test_img,ref_img)
#remove the working directory
shutil.rmtree(working_dir,ignore_errors=True)
print('Done')
