#imports
import os, pathlib, shutil
import numpy as np
from astropath.utilities.config import CONST as UNIV_CONST
from astropath.utilities.img_file_io import get_raw_as_hwl, read_image_from_layer_files
from astropath.shared.samplemetadata import MetadataSummary
from astropath.shared.sample import ReadRectanglesIm3FromXML, XMLLayoutReaderTissue
from astropath.hpfs.flatfield.config import CONST
from astropath.hpfs.flatfield.appliedflatfieldcohort import AppliedFlatfieldCohort
from .testbase import compare_two_csv_files, TestBaseCopyInput, TestBaseSaveOutput

folder = pathlib.Path(__file__).parent
dims = (1004,1344,35)
root = folder/'data'
shardedim3root = folder/'test_for_jenkins'/'applied_flatfield_cohort'/'raw'
slideID = 'M21_1'
rectangle_ns_with_raw_files = [17,18,19,20,23,24,25,26,29,30,31,32,35,36,37,38,39,40]

class DummySample(ReadRectanglesIm3FromXML, XMLLayoutReaderTissue) :

    def __init__(self,*args,filetype='raw',**kwargs) :
        super().__init__(*args,filetype=filetype,uselogfiles=False,**kwargs)

    def run(self,**kwargs) :
        pass

    @classmethod
    def logmodule(cls) : 
        return "dummy_sample"

class TestAppliedFlatfieldCohort(TestBaseCopyInput,TestBaseSaveOutput) :
    """
    Class to test AppliedFlatfieldCohort functions
    """

    @classmethod
    def filestocopy(cls):
        origraw = folder/'data'/'raw'
        for fp in (origraw/slideID).glob('*.Data.dat') :
            yield fp,(shardedim3root/slideID)

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
        if not (shardedim3root/slideID).is_dir() :
            (shardedim3root/slideID).mkdir(parents=True)
        sample = DummySample(root,root/'raw',slideID)
        existing_filepaths = [root/'raw'/slideID/r.file.with_suffix(UNIV_CONST.RAW_EXT) for r in sample.rectangles if r.n in rectangle_ns_with_raw_files]
        for ir,r in enumerate(sample.rectangles) :
            thisrfilepath = shardedim3root/slideID/r.file.with_suffix(UNIV_CONST.RAW_EXT)
            if not thisrfilepath.is_file() :
                shutil.copy(existing_filepaths[ir%len(existing_filepaths)],thisrfilepath)
                self.__files_to_remove.append(thisrfilepath)

    def test_applied_flatfield_cohort(self) :
        #run the cohort
        args = [os.fspath(root),os.fspath(folder/'test_for_jenkins'/'applied_flatfield_cohort'),
                '--shardedim3root',os.fspath(shardedim3root),
                '--sampleregex',slideID,
                '--image-set-split','sequential',
                '--skip-masking',
                '--debug',
               ]
        args.append('--allow-local-edits')
        args.append('--ignore-dependencies')
        AppliedFlatfieldCohort.runfromargumentparser(args=args)
        #compare the output to the reference files
        reffolder = folder/'data'/'reference'/'appliedflatfieldcohort'
        try :
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
            if fp_to_remove.is_file() :
                fp_to_remove.unlink()
        super().tearDown()
