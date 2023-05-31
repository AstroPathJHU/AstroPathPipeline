#imports
import os, pathlib, shutil
import numpy as np
from astropath.utilities.config import CONST as UNIV_CONST
from astropath.utilities.img_file_io import read_image_from_layer_files, write_image_to_file
from astropath.utilities.tableio import readtable
from astropath.hpfs.flatfield.config import CONST
from astropath.hpfs.flatfield.utilities import ComparisonTableEntry
from astropath.hpfs.flatfield.meanimagecomparison import MeanImageComparison
from .testbase import TestBaseCopyInput, TestBaseSaveOutput

folder = pathlib.Path(__file__).parent
dims = (1004,1344,35)
version = 'TEST'
slide_IDs = ['M21_1','M148','M206']

class TestMeanImageComparison(TestBaseCopyInput,TestBaseSaveOutput) :
    """
    Class to test the "meanimagecomparison" code
    """

    workingdir = folder/'test_for_jenkins'/'meanimagecomparison'

    @property
    def outputfilenames(self) :
        all_fps = []
        all_fps.append(self.workingdir/'meanimage_comparison_average_over_all_layers.png')
        all_fps.append(self.workingdir/'meanimagecomparison_table.csv')
        all_fps.append(self.workingdir/'meanimagecomparison.log')
        return all_fps

    @classmethod
    def filestocopy(cls):
        """
        Need to copy the sampledef.csv, Parameters.xml, and Full.xml files and the Scan directories
        """
        newroot = folder/'test_for_jenkins'/'meanimagecomparison'/'root'
        yield folder/'data'/'sampledef.csv', newroot
        for SlideID in slide_IDs :
            newxml=newroot/SlideID/'im3'/'xml'
            parametersfile=folder/'data'/SlideID/'im3'/'xml'/f'{SlideID}.Parameters.xml'
            if parametersfile.exists() :
                yield parametersfile,newxml
            fullfile=folder/'data'/SlideID/'im3'/'xml'/f'{SlideID}.Full.xml'
            if fullfile.exists() :
                yield fullfile,newxml
            newscan=newroot/SlideID/'im3'/'Scan1'
            scandir=folder/'data'/SlideID/'im3'/'Scan1'
            for fp in scandir.glob('*') :
                if not fp.is_dir() :
                    yield fp,newscan
            newMSI=newroot/SlideID/'im3'/'Scan1'/'MSI'
            msidir=folder/'data'/SlideID/'im3'/'Scan1'/'MSI'
            for fp in msidir.glob('*') :
                yield fp,newMSI

    def setUp(self) :
        super().setUp()
        #start a list of filepaths to remove during teardown
        self.__files_to_remove = []
        #create some contrived images to read from the sample subdirectories
        mip = folder/'data'/'reference'/'meanimage'/f'M21_1-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}'
        ref_mia = read_image_from_layer_files(mip,*dims,np.float64)
        ref_semia = 0.05*ref_mia+0.0001
        ref_mia_2 = np.empty_like(ref_mia)
        ref_semia_2 = np.empty_like(ref_semia)
        ref_mia_3 = np.empty_like(ref_mia)
        ref_semia_3 = np.empty_like(ref_semia)
        rand_layer_is = [13, 11, 31, 1, 22, 2, 30, 9, 14, 5, 16, 25, 24, 0, 19, 10, 17, 27, 15, 26, 29, 8, 34, 33, 21, 23, 18, 32, 4, 12, 3, 7, 20, 28, 6]
        for ili,li in enumerate(rand_layer_is) :
            ref_mia_2[:,:,ili] = ref_mia[::-1,:,-(li+1)]
            ref_semia_2[:,:,ili] = ref_semia[::-1,:,-(li+1)]
            ref_mia_3[:,:,ili] = ref_mia[:,:,li]
            ref_semia_3[:,:,ili] = ref_semia[:,:,li]
        ref_mias = [ref_mia,ref_mia_2,ref_mia_3]
        ref_semias = [ref_semia,ref_semia_2,ref_semia_3]
        ref_ms = np.zeros(ref_mia.shape,dtype=np.uint64)+250
        ref_mss = [ref_ms,ref_ms.copy(),ref_ms.copy()]
        #write/copy the files into the samples' meanimage directories
        for sid,mia,semia,ms in zip(slide_IDs,ref_mias,ref_semias,ref_mss) :
            slide_mif = folder/'test_for_jenkins'/'meanimagecomparison'/'root'/sid/UNIV_CONST.IM3_DIR_NAME/'meanimage'
            if not slide_mif.is_dir() :
                slide_mif.mkdir(parents=True)
            write_image_to_file(mia,slide_mif/f'{sid}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            self.__files_to_remove.append(slide_mif/f'{sid}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            write_image_to_file(semia,slide_mif/f'{sid}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            self.__files_to_remove.append(slide_mif/f'{sid}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}')
            write_image_to_file(ms,slide_mif/f'{sid}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}')
            self.__files_to_remove.append(slide_mif/f'{sid}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}')

    def test_meanimage_comparison(self) :
        #run meanimagecomparison selecting the three contrived samples
        root = folder/'test_for_jenkins'/'meanimagecomparison'/'root'
        sampleregex = '('
        for sid in slide_IDs :
            sampleregex+=f'{sid}|'
        sampleregex = sampleregex[:-1]+')'
        args = [os.fspath(root),
                '--sampleregex',sampleregex,
                '--workingdir',os.fspath(self.workingdir),
               ]
        MeanImageComparison.run(args=args)
        #compare the output files with the references
        reffolder = folder/'data'/'reference'/'meanimagecomparison'
        try :
            created = readtable(self.workingdir/'meanimagecomparison_table.csv',ComparisonTableEntry)
            ref = readtable(reffolder/'meanimagecomparison_table.csv',ComparisonTableEntry)
            assert len(created)==len(ref)
            for c in created : 
                r_entry = None
                for r in ref :
                    if ( (c.slide_ID_1 == r.slide_ID_1 and c.slide_ID_2 == r.slide_ID_2) or 
                         (c.slide_ID_2 == r.slide_ID_1 and c.slide_ID_1 == r.slide_ID_2) ) :
                        if c.layer_n==r.layer_n :
                            r_entry = r
                            break
                if r_entry is None :
                    raise RuntimeError(f'ERROR: no reference table entry found for created entry {c}')
                assert c.delta_over_sigma_std_dev == r_entry.delta_over_sigma_std_dev
        except :
            self.saveoutput()
            raise
        else :
            self.removeoutput()
            shutil.rmtree(self.workingdir)

    def tearDown(self) :
        for fp_to_remove in self.__files_to_remove :
            if fp_to_remove.exists() :
                fp_to_remove.unlink()
        super().tearDown()
