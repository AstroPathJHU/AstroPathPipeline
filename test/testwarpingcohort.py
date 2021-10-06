#imports
import pathlib, os, shutil
import numpy as np
from astropath.utilities.config import CONST as UNIV_CONST
from astropath.utilities.img_file_io import read_image_from_layer_files, write_image_to_file
from astropath.shared.samplemetadata import MetadataSummary
from astropath.shared.sample import ReadRectanglesIm3FromXML
from astropath.hpfs.warping.utilities import FieldLog
from astropath.hpfs.warping.warpingcohort import WarpingCohort
from .testbase import compare_two_csv_files, TestBaseSaveOutput

folder = pathlib.Path(__file__).parent
dims = (1004,1344,35)
root = folder/'data'
root2 = folder/'data'/'raw'
slideID = 'M21_1'
et_offset_file = folder/'data'/'corrections'/'best_exposure_time_offsets_Vectra_9_8_2020.csv'
ff_file = folder/'data'/'reference'/'batchflatfieldcohort'/'flatfield_BatchID_99.bin'
rectangle_ns_with_raw_files = [17,18,19,20,23,24,25,26,29,30,31,32,35,36,37,38,39,40]

class DummySample(ReadRectanglesIm3FromXML) :

    def __init__(self,*args,filetype='raw',**kwargs) :
        super().__init__(*args,filetype=filetype,uselogfiles=False,**kwargs)

    def run(self,**kwargs) :
        pass

    @classmethod
    def logmodule(cls) : 
        return "dummy_sample"

class TestWarpingCohort(TestBaseSaveOutput) :
    """
    Class to test WarpingCohort functions
    """

    @classmethod
    def setUpClass(cls) :
        #put together a flatfield file from the individual example layer files
        ff_img = read_image_from_layer_files(ff_file,*(dims),dtype=np.float64)
        (root/UNIV_CONST.FLATFIELD_DIRNAME).mkdir(exist_ok=True,parents=True)
        write_image_to_file(ff_img,root/UNIV_CONST.FLATFIELD_DIRNAME/ff_file.name)
        #move the example background thresholds file to the expected location
        existing_path = folder/'data'/'reference'/'meanimage'/f'{slideID}-background_thresholds.csv'
        new_path = folder/'data'/slideID/UNIV_CONST.IM3_DIR_NAME/'meanimage'/existing_path.name
        shutil.copy(existing_path,new_path)
        super().setUpClass()

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

    @property
    def output_dir(self) :
        return folder/'test_for_jenkins'/'warping_cohort'

    @property
    def outputfilenames(self) :
        all_fps = []
        all_fps.append(self.output_dir/'octets'/f'{slideID}-all_overlap_octets.csv')
        all_fps.append(self.output_dir/'initial_pattern_fit_results_2_layer_1.csv')
        all_fps.append(self.output_dir/'initial_pattern_fits_field_logs.csv')
        all_fps.append(self.output_dir/'initial_pattern_fits_metadata_summaries.csv')
        all_fps.append(self.output_dir/'principal_point_fit_results_2_layer_1.csv')
        all_fps.append(self.output_dir/'principal_point_fits_field_logs.csv')
        all_fps.append(self.output_dir/'principal_point_fits_metadata_summaries.csv')
        all_fps.append(self.output_dir/'final_pattern_fit_results_2_layer_1.csv')
        all_fps.append(self.output_dir/'final_pattern_fits_field_logs.csv')
        all_fps.append(self.output_dir/'final_pattern_fits_metadata_summaries.csv')
        all_fps.append(self.output_dir/'weighted_average_warp.csv')
        return all_fps

    def test_warping_cohort_octets_only(self) :
        #run the cohort
        args = [os.fspath(root),os.fspath(root2),
                '--exposure-time-offset-file',os.fspath(et_offset_file),
                '--flatfield-file',ff_file.name,
                '--sampleregex',slideID,
                '--workingdir',os.fspath(self.output_dir),
                '--octets-only',
               ]
        args.append('--allow-local-edits')
        args.append('--ignore-dependencies')
        WarpingCohort.runfromargumentparser(args=args)
        #just make sure that the empty octet output file exists
        try :
            self.assertTrue((self.output_dir/'octets'/f'{slideID}-all_overlap_octets.csv').is_file())
        except :
            self.saveoutput()
            raise
        else :
            self.removeoutput()
            shutil.rmtree(self.output_dir)

    def test_warping_cohort(self) :
        #first we need to copy the contrived octet and octet split files to the output directory
        existing_paths = []
        existing_paths.append(folder/'data'/'reference'/'warpingcohort'/f'{slideID}-all_overlap_octets.csv')
        existing_paths.append(folder/'data'/'reference'/'warpingcohort'/'initial_pattern_octets_selected.csv')
        existing_paths.append(folder/'data'/'reference'/'warpingcohort'/'principal_point_octets_selected.csv')
        existing_paths.append(folder/'data'/'reference'/'warpingcohort'/'final_pattern_octets_selected.csv')
        for existing_path in existing_paths :
            if not (self.output_dir/'octets').is_dir() :
                (self.output_dir/'octets').mkdir(parents=True)
            new_path = self.output_dir/'octets'/existing_path.name
            shutil.copy(existing_path,new_path)
        #run the cohort
        args = [os.fspath(root),os.fspath(root2),
                '--exposure-time-offset-file',os.fspath(et_offset_file),
                '--flatfield-file',ff_file.name,
                '--sampleregex',slideID,
                '--workingdir',os.fspath(self.output_dir),
                '--initial-pattern-octets','2',
                '--principal-point-octets','2',
                '--final-pattern-octets','2',
               ]
        args.append('--allow-local-edits')
        args.append('--ignore-dependencies')
        WarpingCohort.runfromargumentparser(args=args)
        #after running we can remove the octet and octet split files
        for existing_path in existing_paths :
            (self.output_dir/'octets'/existing_path.name).unlink()
        #make sure the fit result files all exist and that the other files match the references
        reffolder = folder/'data'/'reference'/'warpingcohort'
        try :
            for ofp in self.outputfilenames :
                if 'octets' in ofp.name :
                    continue
                if 'fit_results' in ofp.name or ofp.name=='weighted_average_warp.csv' :
                    self.assertTrue(ofp.is_file())
                else :
                    if 'field_logs' in ofp.name :
                        compare_two_csv_files(ofp.parent,reffolder,ofp.name,FieldLog)
                    elif 'metadata_summaries' in ofp.name :
                        compare_two_csv_files(ofp.parent,reffolder,ofp.name,MetadataSummary)
                    else :
                        raise ValueError(f'ERROR: unknown output file {ofp}')
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

    @classmethod
    def tearDownClass(cls) :
        #Remove the flatfield file that was created and the directory it's in
        shutil.rmtree(root/UNIV_CONST.FLATFIELD_DIRNAME)
        #Remove the copied background threshold file
        (folder/'data'/slideID/UNIV_CONST.IM3_DIR_NAME/UNIV_CONST.MEANIMAGE_DIRNAME/f'{slideID}-background_thresholds.csv').unlink()
        super().tearDownClass()
