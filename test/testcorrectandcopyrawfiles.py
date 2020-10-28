#imports
from ..correct_and_copy_rawfiles.rawfile_corrector import RawfileCorrector
from ..utilities.img_file_io import getRawAsHWL, getRawAsHW
from ..utilities.misc import cd
import numpy as np
from argparse import Namespace
import pathlib, glob, shutil

#some constants
folder = pathlib.Path(__file__).parent
workingdir_name = 'correct_and_copy_rawfiles_test_for_jenkins'
working_dir = folder/workingdir_name
working_dir.mkdir(exist_ok=True)
multilayer_ref_path = folder/'reference'/'correctandcopyrawfiles'/'multilayer'
singlelayer_ref_path = folder/'reference'/'correctandcopyrawfiles'/'singlelayer'
dims = (1004,1344,35)


#start by running for all layers at once
print('TESTING MULTILAYER CORRECT/COPY')
args = Namespace(
        sample='M21_1',
        rawfile_top_dir=str(folder/'data'/'raw'),
        metadata_top_dir=str(folder/'data'),
        workingdir=str(working_dir),
        exposure_time_offset_file=str(folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'),
        skip_exposure_time_correction=False,
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
corrector = RawfileCorrector(args)
corrector.run()
#check the xmls
with cd(working_dir) :
    xml_filenames = glob.glob('*.xml')
for xmlfn in xml_filenames :
    with open(working_dir/xmlfn,'r') as fp :
        test_lines=fp.readlines()
    with open(multilayer_ref_path/xmlfn,'r') as fp :
        ref_lines=fp.readlines()
    for testline,refline in zip(test_lines,ref_lines) :
        if testline!=refline :
            raise RuntimeError(f'ERROR: xml file {xmlfn} lines are not identical!\ntest line = {testline}ref. line={refline}')
#check the output image files
with cd(working_dir) :
    img_filenames = glob.glob('*.fw')
for imfn in img_filenames :
    test_img = getRawAsHWL(working_dir/imfn,dims[0],dims[1],dims[2])
    ref_img = getRawAsHWL(multilayer_ref_path/imfn,dims[0],dims[1],dims[2])
    np.testing.assert_array_equal(test_img,ref_img)
#remove the working directory
shutil.rmtree(working_dir,ignore_errors=True)

#run again with only the first layer
print('TESTING SINGLE LAYER CORRECT/COPY')
working_dir.mkdir(exist_ok=True)
args = Namespace(
        sample='M21_1',
        rawfile_top_dir=str(folder/'data'/'raw'),
        metadata_top_dir=str(folder/'data'),
        workingdir=str(working_dir),
        exposure_time_offset_file=str(folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'),
        skip_exposure_time_correction=False,
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
corrector = RawfileCorrector(args)
corrector.run()
#check the xmls
with cd(working_dir) :
    xml_filenames = glob.glob('*.xml')
for xmlfn in xml_filenames :
    with open(working_dir/xmlfn,'r') as fp :
        test_lines=fp.readlines()
    with open(singlelayer_ref_path/xmlfn,'r') as fp :
        ref_lines=fp.readlines()
    for testline,refline in zip(test_lines,ref_lines) :
        if testline!=refline :
            raise RuntimeError(f'ERROR: xml file {xmlfn} lines are not identical!\ntest line = {testline}ref. line={refline}')
#check the output image files
with cd(working_dir) :
    img_filenames = glob.glob('*.fw01')
for imfn in img_filenames :
    test_img = getRawAsHW(working_dir/imfn,dims[0],dims[1])
    ref_img = getRawAsHW(singlelayer_ref_path/imfn,dims[0],dims[1])
    np.testing.assert_array_equal(test_img,ref_img)
#remove the working directory
shutil.rmtree(working_dir,ignore_errors=True)
print('Done')
