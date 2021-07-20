#imports
from astropath.hpfs.flatfield.appliedflatfieldcohort import AppliedFlatfieldCohort
from astropath.shared.sample import ReadRectanglesIm3FromXML
from astropath.hpfs.flatfield.utilities import FieldLog
from astropath.hpfs.flatfield.config import CONST
from astropath.shared.samplemetadata import MetadataSummary
from astropath.utilities.img_file_io import get_raw_as_hwl, read_image_from_layer_files
from astropath.utilities.config import CONST as UNIV_CONST
from .testbase import compare_two_csv_files, TestBaseSaveOutput
import numpy as np
import os, pathlib, shutil

folder = pathlib.Path(__file__).parent
dims = (1004,1344,35)
root = folder/'data'
root2 = folder/'data'/'raw'
et_offset_file = folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'
slideID = 'M21_1'
rectangle_ns_with_raw_files = [17,18,19,20,23,24,25,26,29,30,31,32,35,36,37,38,39,40]

class DummySample(ReadRectanglesIm3FromXML) :

    def __init__(self,*args,filetype='raw',**kwargs) :
        super().__init__(*args,filetype=filetype,uselogfiles=False,**kwargs)

    def run(self,**kwargs) :
        pass

    @classmethod
    def logmodule(cls) : 
        return "dummy_sample"

class TestAppliedFlatfieldCohort(TestBaseSaveOutput) :
    """
    Class to test AppliedFlatfieldCohort functions
    """

    @property
    def output_dir(self) :
        return folder/'test_for_jenkins'/'applied_flatfield_cohort'

    @property
    def outputfilenames(self) :
        all_fps = []
        all_fps.append(self.output_dir/(CONST.FIELDS_USED_CSV_FILENAME.rstrip('.csv')+'_flatfield.csv'))
        all_fps.append(self.output_dir/(CONST.FIELDS_USED_CSV_FILENAME.rstrip('.csv')+'_corrected_mean_image.csv'))
        all_fps.append(self.output_dir/'metadata_summary_flatfield_stacked_images.csv')
        all_fps.append(self.output_dir/'metadata_summary_corrected_mean_image_stacked_images.csv')
        all_fps.append(self.output_dir/'flatfield.bin')
        all_fps.append(self.output_dir/'flatfield_uncertainty.bin')
        all_fps.append(self.output_dir/'corrected_mean_image.bin')
        all_fps.append(self.output_dir/'corrected_mean_image_uncertainty.bin')
        return all_fps

    def setUp(self) :
        """
        Need to contrive some extra raw data files to have enough to do the test
        Will copy some that already exist to do that 
        """
        super().setUp()
        self.__files_to_remove = []
        sample = DummySample(root,root2,slideID)
        existing_filepaths = [root2/slideID/r.file.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.RAW_EXT) for r in sample.rectangles if r.n in rectangle_ns_with_raw_files]
        for ir,r in enumerate(sample.rectangles) :
            thisrfilepath = root2/slideID/r.file.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.RAW_EXT)
            if not thisrfilepath.is_file() :
                shutil.copy(existing_filepaths[ir%len(existing_filepaths)],thisrfilepath)
                self.__files_to_remove.append(thisrfilepath)

    def test_applied_flatfield_cohort(self) :
        #run the cohort
        args = [os.fspath(root),os.fspath(root2),os.fspath(folder/'test_for_jenkins'/'applied_flatfield_cohort'),
                '--exposure_time_offset_file',os.fspath(et_offset_file),
                '--sampleregex',slideID,
                '--image_set_split','sequential',
               ]
        args.append('--allow-local-edits')
        args.append('--ignore-dependencies')
        AppliedFlatfieldCohort.runfromargumentparser(args=args)
        #compare the output to the reference files
        reffolder = folder/'data'/'reference'/'appliedflatfieldcohort'
        try :
            compare_two_csv_files(self.output_dir,reffolder,f"{CONST.FIELDS_USED_CSV_FILENAME.rstrip('.csv')}_flatfield.csv",FieldLog)
            compare_two_csv_files(self.output_dir,reffolder,f"{CONST.FIELDS_USED_CSV_FILENAME.rstrip('.csv')}_corrected_mean_image.csv",FieldLog)
            compare_two_csv_files(self.output_dir,reffolder,'metadata_summary_flatfield_stacked_images.csv',MetadataSummary)
            compare_two_csv_files(self.output_dir,reffolder,'metadata_summary_corrected_mean_image_stacked_images.csv',MetadataSummary)
            ffa = get_raw_as_hwl(self.output_dir/'flatfield.bin',*dims,np.float64)
            ref_ffa = read_image_from_layer_files(reffolder/'flatfield.bin',*dims,np.float64)
            np.testing.assert_allclose(ffa,ref_ffa,rtol=1e-09)
            cmia = get_raw_as_hwl(self.output_dir/'corrected_mean_image.bin',*dims,np.float64)
            ref_cmia = read_image_from_layer_files(reffolder/'corrected_mean_image.bin',*dims,np.float64)
            np.testing.assert_allclose(cmia,ref_cmia,rtol=1e-09)
        except :
            self.saveoutput()
            raise
        else :
            self.removeoutput()
            shutil.rmtree(self.output_dir)

    def tearDown(self) :
        for fp_to_remove in self.__files_to_remove :
            fp_to_remove.unlink()
        super().tearDown()
