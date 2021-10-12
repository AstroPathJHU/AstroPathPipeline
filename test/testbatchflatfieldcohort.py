#imports
from astropath.hpfs.flatfield.batchflatfieldcohort import BatchFlatfieldMultiCohort
from astropath.hpfs.flatfield.utilities import FieldLog
from astropath.hpfs.flatfield.config import CONST
from astropath.shared.samplemetadata import MetadataSummary
from astropath.utilities.img_file_io import get_raw_as_hwl, read_image_from_layer_files, write_image_to_file
from astropath.utilities.config import CONST as UNIV_CONST
from .testbase import compare_two_csv_files, TestBaseSaveOutput
import numpy as np
import os, pathlib, shutil

folder = pathlib.Path(__file__).parent
dims = (1004,1344,35)
batchID = 99
slide_IDs = ['M21_1','M148','M206']

class TestBatchFlatfieldCohort(TestBaseSaveOutput) :
    """
    Class to test BatchFlatfieldCohort functions
    """

    @property
    def batchflatfield_dir(self) :
        return folder/'data'/UNIV_CONST.FLATFIELD_DIRNAME/f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}'

    @property
    def outputfilenames(self) :
        all_fps = []
        all_fps.append(self.batchflatfield_dir/CONST.FIELDS_USED_CSV_FILENAME)
        all_fps.append(self.batchflatfield_dir/f'{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}')
        all_fps.append(self.batchflatfield_dir/f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}_uncertainty.bin')
        all_fps.append(self.batchflatfield_dir.parent/f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}.bin')
        return all_fps

    def setUp(self) :
        super().setUp()
        #start a list of filepaths to remove during teardown
        self.__files_to_remove = []
        #create some contrived images to read from the sample subdirectories
        ref_mia = read_image_from_layer_files(folder/'data'/'reference'/'meanimage'/f'M21_1-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}',*dims,np.float64)
        ref_msa = read_image_from_layer_files(folder/'data'/'reference'/'meanimage'/f'M21_1-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}',*dims,np.uint64)
        ref_sisa = read_image_from_layer_files(folder/'data'/'reference'/'meanimage'/f'M21_1-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}',*dims,np.float64)
        ref_mia_2 = ref_mia[:,:,::-1]
        ref_msa_2 = ref_msa[:,:,::-1]
        ref_sisa_2 = ref_sisa[:,:,::-1]
        ref_mia_3 = np.empty_like(ref_mia)
        ref_msa_3 = np.empty_like(ref_msa)
        ref_sisa_3 = np.empty_like(ref_sisa)
        for ili,li in enumerate([17,28,32,2,15,7,10,27,24,21,18,0,33,23,13,9,30,19,12,25,3,8,14,20,16,1,11,34,4,29,5,31,22,26,6]) :
            ref_mia_3[ili] = ref_mia[li]
            ref_msa_3[ili] = ref_msa[li]
            ref_sisa_3[ili] = ref_sisa[li]
        ref_mias = [ref_mia,ref_mia_2,ref_mia_3]
        ref_msas = [ref_msa,ref_msa_2,ref_msa_3]
        ref_sisas = [ref_sisa,ref_sisa_2,ref_sisa_3]
        #will also copy some of the metadata info that's needed
        ref_fields_used = folder/'data'/'reference'/'meanimage'/CONST.FIELDS_USED_CSV_FILENAME
        ref_metadata_summary = folder/'data'/'reference'/'meanimage'/f'M21_1-{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}'
        #write/copy the files into the samples' meanimage directories
        for sid,mia,msa,sisa in zip(slide_IDs,ref_mias,ref_msas,ref_sisas) :
            slide_meanimage_folder = folder/'data'/sid/UNIV_CONST.IM3_DIR_NAME/'meanimage'
            if not slide_meanimage_folder.is_dir() :
                slide_meanimage_folder.mkdir(parents=True)
            write_image_to_file(mia,slide_meanimage_folder/f'{sid}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            self.__files_to_remove.append(slide_meanimage_folder/f'{sid}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            write_image_to_file(msa,slide_meanimage_folder/f'{sid}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}')
            self.__files_to_remove.append(slide_meanimage_folder/f'{sid}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}')
            write_image_to_file(sisa,slide_meanimage_folder/f'{sid}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}')
            self.__files_to_remove.append(slide_meanimage_folder/f'{sid}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}')
            shutil.copy(ref_fields_used,slide_meanimage_folder)
            self.__files_to_remove.append(slide_meanimage_folder/ref_fields_used.name)
            shutil.copy(ref_metadata_summary,slide_meanimage_folder/f'{sid}-{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}')
            self.__files_to_remove.append(slide_meanimage_folder/f'{sid}-{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}')

    def test_batch_flatfield_cohort(self) :
        #run the BatchFlatfieldCohort selecting the three contrived samples
        root = folder/'data'
        shardedim3root = folder/'data'/'raw'
        args = [os.fspath(root),
                '--shardedim3root',os.fspath(shardedim3root),
                '--sampleregex','('+'|'.join(slide_IDs)+')',
                '--batchID',str(batchID),
                '--allow-local-edits',
                '--ignore-dependencies',
               ]
        BatchFlatfieldMultiCohort.runfromargumentparser(args=args)
        #compare the output files with the references
        reffolder = folder/'data'/'reference'/'batchflatfieldcohort'
        try :
            compare_two_csv_files(self.batchflatfield_dir,reffolder,CONST.FIELDS_USED_CSV_FILENAME,FieldLog)
            compare_two_csv_files(self.batchflatfield_dir,reffolder,f'{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}',MetadataSummary)
            ffa = get_raw_as_hwl(self.batchflatfield_dir.parent/f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}.bin',*dims,np.float64)
            ref_ffa = read_image_from_layer_files(reffolder/f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}.bin',*dims,np.float64)
            np.testing.assert_allclose(ffa,ref_ffa,rtol=1e-09)
            ffua = get_raw_as_hwl(self.batchflatfield_dir/f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}_uncertainty.bin',*dims,np.float64)
            ref_ffua = read_image_from_layer_files(reffolder/f'{CONST.FLATFIELD_DIRNAME_STEM}{batchID:02d}_uncertainty.bin',*dims,np.float64)
            np.testing.assert_allclose(ffua,ref_ffua,rtol=1e-09)
        except :
            self.saveoutput()
            raise
        else :
            self.removeoutput()
            shutil.rmtree(self.batchflatfield_dir)

    def tearDown(self) :
        for fp_to_remove in self.__files_to_remove :
            fp_to_remove.unlink()
        super().tearDown()

