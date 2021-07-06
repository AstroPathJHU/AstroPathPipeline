#imports
#from astropath.hpfs.flatfield.meanimagesample import MeanImageSample
from astropath.hpfs.flatfield.meanimagecohort import MeanImageCohort
from astropath.hpfs.flatfield.config import CONST
from astropath.utilities.config import CONST as UNIV_CONST
from .testbase import TestBaseSaveOutput
import os, pathlib

folder = pathlib.Path(__file__).parent
SlideID = 'M21_1'
rectangle_files_with_full_masks = [
    'M21_1_[45093,14453]',
    'M21_1_[45093,14853]',
    'M21_1_[45628,14853]',
    'M21_1_[46163,14053]',

]

class TestMeanImage(TestBaseSaveOutput) :
    """
    Class to test MeanImageSample and MeanImageCohort functions
    """

    @property
    def outputfilenames(self) :
        meanimage_dir = folder/'data'/SlideID/'im3'/UNIV_CONST.MEANIMAGE_DIRNAME
        all_fps = []
        all_fps.append(meanimage_dir/CONST.FIELDS_USED_CSV_FILENAME)
        all_fps.append(meanimage_dir/f'{SlideID}-{CONST.BACKGROUND_THRESHOLD_CSV_FILE_NAME_STEM}')
        all_fps.append(meanimage_dir/f'{SlideID}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}')
        all_fps.append(meanimage_dir/f'{SlideID}-{CONST.MASKING_SUMMARY_PDF_FILENAME}')
        all_fps.append(meanimage_dir/f'{SlideID}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}')
        all_fps.append(meanimage_dir/f'{SlideID}-{CONST.MEANIMAGE_SUMMARY_PDF_FILENAME}')
        all_fps.append(meanimage_dir/f'{SlideID}-{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}')
        all_fps.append(meanimage_dir/f'{SlideID}-{CONST.METADATA_SUMMARY_THRESHOLDING_IMAGES_CSV_FILENAME}')
        all_fps.append(meanimage_dir/f'{SlideID}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}')
        all_fps.append(meanimage_dir/f'{SlideID}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}')
        all_fps.append(meanimage_dir/f'{SlideID}-{CONST.THRESHOLDING_DATA_TABLE_CSV_FILENAME}')
        all_fps.append(meanimage_dir/f'{SlideID}-{CONST.THRESHOLDING_SUMMARY_PDF_FILENAME}')
        masking_dir = meanimage_dir/CONST.IMAGE_MASKING_SUBDIR_NAME
        all_fps.append(masking_dir/CONST.LABELLED_MASK_REGIONS_CSV_FILENAME)
        for fn in (folder/'data'/'raw'/SlideID).glob(f'*{UNIV_CONST.RAW_EXT}') :
            all_fps.append(masking_dir/f'{fn.name.rstrip(UNIV_CONST.RAW_EXT)}_{CONST.TISSUE_MASK_FILE_NAME_STEM}')
        for fns in rectangle_files_with_full_masks :
            all_fps.append(masking_dir/f'{fns}_{CONST.BLUR_AND_SATURATION_MASK_FILE_NAME_STEM}')
        return all_fps

    def test_mean_image(self,n_threads=10) :
        root = folder/'data'
        root2 = folder/'data'/'raw'
        et_offset_file = folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'
        args = [os.fspath(root),os.fspath(root2),
                '--exposure_time_offset_file',os.fspath(et_offset_file),
                '--njobs',str(n_threads),
                '--sampleregex',SlideID,
                '--allow-local-edits',
               ]
        MeanImageCohort.runfromargumentparser(args=args)
        self.saveoutput()
