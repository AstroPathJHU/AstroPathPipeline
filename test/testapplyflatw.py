#imports
import os, pathlib, shutil
import numpy as np
from astropath.utilities.config import CONST as UNIV_CONST
from astropath.utilities.img_file_io import get_raw_as_hwl, get_raw_as_hw
from astropath.utilities.img_file_io import read_image_from_layer_files, write_image_to_file
from astropath.hpfs.imagecorrection.applyflatwcohort import ApplyFlatWCohort
from .testbase import TestBaseCopyInput, TestBaseSaveOutput

folder = pathlib.Path(__file__).parent
dims = (1004,1344,35)
version = 'TEST'
warping_correction_filepath = folder/'data'/'corrections'/'TEST_WARPING_weighted_average_warp.csv'
slide_ID = 'M21_1'
rectangle_ns_with_raw_files = [17,18,19,20,23,24,25,26,29,30,31,32,35,36,37,38,39,40]
shardedim3root = folder/'data'/'raw'

class TestApplyFlatWCohort(TestBaseCopyInput, TestBaseSaveOutput) :
    """
    Class to test ApplyFlatWCohort functions
    """

    @classmethod
    def filestocopy(cls) :
        oldroot = folder/'data'
        newroot = folder/'test_for_jenkins'/'applyflatw'/'root'
        for fp in oldroot.glob('*') :
            if fp.is_dir() :
                continue
            yield fp, newroot

    @property
    def outputfilenames(self) :
        all_fps = []
        for fns in [fp.name.rstrip(UNIV_CONST.RAW_EXT) for fp in (shardedim3root/slide_ID).glob(f'*{UNIV_CONST.RAW_EXT}')] :
            all_fps.append(folder/'test_for_jenkins'/'applyflatw'/'flatw'/f'{fns}{UNIV_CONST.FLATW_EXT}')
            all_fps.append(folder/'test_for_jenkins'/'applyflatw'/'flatw'/f'{fns}{UNIV_CONST.FLATW_EXT}01')
        return all_fps

    def setUp(self) :
        super().setUp()
        self.__files_to_remove = []
        #Mock up a flatfield file to use for applying corrections
        ff = read_image_from_layer_files(folder/'data'/'reference'/'batchflatfieldcohort'/f'{UNIV_CONST.FLATFIELD_DIRNAME}_{version}.bin',*dims,np.float64)
        write_image_to_file(ff,folder/'data'/'corrections'/f'{UNIV_CONST.FLATFIELD_DIRNAME}_{version}.bin')
        self.__files_to_remove.append(folder/'data'/'corrections'/f'{UNIV_CONST.FLATFIELD_DIRNAME}_{version}.bin')
        #Add a slide_ID dir in the fake root directory (will get removed with the rest of the test_for_jenkins directory)
        (folder/'test_for_jenkins'/'applyflatw'/'root'/slide_ID).mkdir(parents=True,exist_ok=True)

    def test_image_correction_cohort(self) :
        #run the image correction cohort
        args = [os.fspath(folder/'test_for_jenkins'/'applyflatw'/'root'),
                '--shardedim3root',os.fspath(shardedim3root),
                '--im3root',os.fspath(folder/'data'),
                '--sampleregex',slide_ID,
                '--workingdir',os.fspath(folder/'test_for_jenkins'/'applyflatw'/'flatw'),
                '--flatfield-file',os.fspath(folder/'data'/'corrections'/f'{UNIV_CONST.FLATFIELD_DIRNAME}_{version}.bin'),
                '--warping-file',os.fspath(warping_correction_filepath),
                '--layers','-1','1',
                '--njobs','1',
                '--allow-local-edits',
                '--ignore-dependencies',
                '--debug',
                '--selectrectangles',
                *(str(rn) for rn in rectangle_ns_with_raw_files),
                ]
        ApplyFlatWCohort.runfromargumentparser(args=args)
        #compare the results to the reference files
        try :
            for fp in (folder/'data'/'reference'/'applyflatw').glob('*') :
                if fp.name.endswith('01') :
                    refa = get_raw_as_hw(fp,*(dims[:-1]),np.uint16)
                    testa = get_raw_as_hw(folder/'test_for_jenkins'/'applyflatw'/'flatw'/slide_ID/fp.name,*(dims[:-1]),np.uint16)
                else :
                    refa = get_raw_as_hwl(fp,*dims,np.uint16)
                    testa = get_raw_as_hwl(folder/'test_for_jenkins'/'applyflatw'/'flatw'/slide_ID/fp.name,*dims,np.uint16)
            np.testing.assert_allclose(testa,refa,rtol=1e-09)
        except :
            self.saveoutput()
            raise
        finally :
            self.removeoutput()
            shutil.rmtree(folder/'test_for_jenkins'/'applyflatw')

    def tearDown(self) :
        for fp_to_remove in self.__files_to_remove :
            fp_to_remove.unlink()
        super().tearDown()
