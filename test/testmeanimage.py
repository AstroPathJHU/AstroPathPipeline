#imports
import os, pathlib, shutil
import numpy as np
from astropath.utilities.config import CONST as UNIV_CONST
from astropath.utilities.img_file_io import get_raw_as_hwl, read_image_from_layer_files
from astropath.shared.samplemetadata import MetadataSummary
from astropath.shared.image_masking.utilities import LabelledMaskRegion
from astropath.hpfs.flatfield.config import CONST
from astropath.hpfs.flatfield.utilities import FieldLog, RectangleThresholdTableEntry, ThresholdTableEntry
from astropath.hpfs.flatfield.meanimagecohort import MeanImageCohort
from .testbase import compare_two_csv_files, TestBaseSaveOutput

folder = pathlib.Path(__file__).parent
dims = (1004,1344,35)
SlideID = 'M21_1'
rectangle_ns_with_raw_files = [17,18,19,20,23,24,25,26,29,30,31,32,35,36,37,38,39,40]
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
    def meanimage_dir(self) :
        return folder/'data'/SlideID/UNIV_CONST.IM3_DIR_NAME/UNIV_CONST.MEANIMAGE_DIRNAME

    @property
    def masking_dir(self) :
        return folder/'test_for_jenkins'/'mean_image'/SlideID/UNIV_CONST.IM3_DIR_NAME/UNIV_CONST.MEANIMAGE_DIRNAME/CONST.IMAGE_MASKING_SUBDIR_NAME

    @property
    def outputfilenames(self) :
        all_fps = []
        all_fps.append(self.meanimage_dir/CONST.FIELDS_USED_CSV_FILENAME)
        all_fps.append(self.meanimage_dir/f'{SlideID}-{CONST.BACKGROUND_THRESHOLD_CSV_FILE_NAME_STEM}')
        all_fps.append(self.meanimage_dir/f'{SlideID}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}')
        all_fps.append(self.meanimage_dir/f'{SlideID}-{CONST.MASKING_SUMMARY_PDF_FILENAME}')
        all_fps.append(self.meanimage_dir/f'{SlideID}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}')
        all_fps.append(self.meanimage_dir/f'{SlideID}-{CONST.MEANIMAGE_SUMMARY_PDF_FILENAME}')
        all_fps.append(self.meanimage_dir/f'{SlideID}-{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}')
        all_fps.append(self.meanimage_dir/f'{SlideID}-{CONST.METADATA_SUMMARY_THRESHOLDING_IMAGES_CSV_FILENAME}')
        all_fps.append(self.meanimage_dir/f'{SlideID}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}')
        all_fps.append(self.meanimage_dir/f'{SlideID}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}')
        all_fps.append(self.meanimage_dir/f'{SlideID}-{CONST.THRESHOLDING_DATA_TABLE_CSV_FILENAME}')
        all_fps.append(self.meanimage_dir/f'{SlideID}-{CONST.THRESHOLDING_SUMMARY_PDF_FILENAME}')
        all_fps.append(self.masking_dir/CONST.LABELLED_MASK_REGIONS_CSV_FILENAME)
        for fn in (folder/'data'/'raw'/SlideID).glob(f'*{UNIV_CONST.RAW_EXT}') :
            all_fps.append(self.masking_dir/f'{fn.name.rstrip(UNIV_CONST.RAW_EXT)}_{CONST.TISSUE_MASK_FILE_NAME_STEM}')
        for fns in rectangle_files_with_full_masks :
            all_fps.append(self.masking_dir/f'{fns}_{CONST.BLUR_AND_SATURATION_MASK_FILE_NAME_STEM}')
        return all_fps

    def test_mean_image(self,n_threads=1) :
        #run the MeanImageCohort selecting just the single sample with raw files
        root = folder/'data'
        shardedim3root = folder/'data'/'raw'
        et_offset_file = folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'
        (folder/'test_for_jenkins'/'mean_image'/SlideID/UNIV_CONST.IM3_DIR_NAME/UNIV_CONST.MEANIMAGE_DIRNAME/CONST.IMAGE_MASKING_SUBDIR_NAME).mkdir(parents=True,exist_ok=True)
        args = [os.fspath(root),
                '--shardedim3root',os.fspath(shardedim3root),
                '--exposure-time-offset-file',os.fspath(et_offset_file),
                '--njobs',str(n_threads),
                '--sampleregex',SlideID,
                '--maskroot',os.fspath(folder/'test_for_jenkins'/'mean_image'),
                '--selectrectangles'
                ]
        for rn in rectangle_ns_with_raw_files :
            args.append(str(rn))
        args.append('--allow-local-edits')
        MeanImageCohort.runfromargumentparser(args=args)
        #compare the output files with the references
        reffolder = folder/'data'/'reference'/'meanimage'
        try :
            compare_two_csv_files(self.meanimage_dir,reffolder,CONST.FIELDS_USED_CSV_FILENAME,FieldLog)
            compare_two_csv_files(self.meanimage_dir,reffolder,f'{SlideID}-{CONST.BACKGROUND_THRESHOLD_CSV_FILE_NAME_STEM}',ThresholdTableEntry)
            compare_two_csv_files(self.meanimage_dir,reffolder,f'{SlideID}-{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}',MetadataSummary)
            compare_two_csv_files(self.meanimage_dir,reffolder,f'{SlideID}-{CONST.METADATA_SUMMARY_THRESHOLDING_IMAGES_CSV_FILENAME}',MetadataSummary)
            compare_two_csv_files(self.meanimage_dir,reffolder,f'{SlideID}-{CONST.THRESHOLDING_DATA_TABLE_CSV_FILENAME}',RectangleThresholdTableEntry)
            compare_two_csv_files(self.masking_dir,reffolder,CONST.LABELLED_MASK_REGIONS_CSV_FILENAME,LabelledMaskRegion)
            msa = get_raw_as_hwl(self.meanimage_dir/f'{SlideID}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}',*dims,np.float64)
            ref_msa = read_image_from_layer_files(reffolder/f'{SlideID}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}',*dims,np.float64)
            np.testing.assert_allclose(msa,ref_msa,rtol=1e-09)
            mia = get_raw_as_hwl(self.meanimage_dir/f'{SlideID}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}',*dims,np.float64)
            ref_mia = read_image_from_layer_files(reffolder/f'{SlideID}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}',*dims,np.float64)
            np.testing.assert_allclose(mia,ref_mia,rtol=1e-09)
            semia = get_raw_as_hwl(self.meanimage_dir/f'{SlideID}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}',*dims,np.float64)
            ref_semia = read_image_from_layer_files(reffolder/f'{SlideID}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}',*dims,np.float64)
            np.testing.assert_allclose(semia,ref_semia,rtol=1e-09)
            sisa = get_raw_as_hwl(self.meanimage_dir/f'{SlideID}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}',*dims,np.float64)
            ref_sisa = read_image_from_layer_files(reffolder/f'{SlideID}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}',*dims,np.float64)
            np.testing.assert_allclose(sisa,ref_sisa,rtol=1e-08)
        except :
            self.saveoutput()
            raise
        else :
            self.removeoutput()
            tpdf = self.meanimage_dir/CONST.THRESHOLDING_SUMMARY_PDF_FILENAME
            if tpdf.is_file() :
                tpdf.unlink()
            td = self.meanimage_dir/(CONST.THRESHOLDING_SUMMARY_PDF_FILENAME.replace('.pdf','_plots'))
            if td.is_dir() :
                shutil.rmtree(td)
            mspdf = self.masking_dir.parent/CONST.MASKING_SUMMARY_PDF_FILENAME
            if mspdf.is_file() :
                mspdf.unlink()
            mispdf = self.meanimage_dir/CONST.MEANIMAGE_SUMMARY_PDF_FILENAME
            if mispdf.is_file() :
                mispdf.unlink()
            misd = self.meanimage_dir/(CONST.MEANIMAGE_SUMMARY_PDF_FILENAME.replace('.pdf','_plots'))
            if misd.is_dir() :
                shutil.rmtree(misd)
            shutil.rmtree(self.masking_dir)

