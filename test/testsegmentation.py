#imports
import os, pathlib, shutil
import numpy as np
from astropath.slides.segmentation.segmentationsample import SegmentationSampleNNUNet, SegmentationSampleDeepCell
from .testbase import TestBaseCopyInput, TestBaseSaveOutput

folder = pathlib.Path(__file__).parent
slide_ID = 'M21_1'
rectangle_ns_with_comp_tiff_files_nnunet = [18,23,40]
rectangle_ns_with_comp_tiff_files_deepcell = [1,17,18,23,40]

class TestSegmentationBase(TestBaseCopyInput, TestBaseSaveOutput) :
    """
    Base class for testing segmentation routines
    """

    @classmethod
    def filestocopy(cls) :
        oldroot = folder/'data'
        newroot = folder/'test_for_jenkins'/'segmentation'/'root'
        for fp in oldroot.glob('*') :
            if fp.is_dir() :
                continue
            yield fp, newroot
        oldcomptiffs = folder/'data'/slide_ID/'inform_data'/'Component_Tiffs'
        newcomptiffs = folder/'test_for_jenkins'/'segmentation'/'root'/slide_ID/'inform_data'/'Component_Tiffs'
        if not newcomptiffs.is_dir() :
            newcomptiffs.mkdir(parents=True)
        for fp in oldcomptiffs.glob('*') :
            yield fp,newcomptiffs
        oldscan1 = folder/'data'/slide_ID/'im3'/'Scan1'
        newscan1 = folder/'test_for_jenkins'/'segmentation'/'root'/slide_ID/'im3'/'Scan1'
        if not newscan1.is_dir() :
            newscan1.mkdir(parents=True)
        for fp in oldscan1.rglob('*') :
            if not fp.is_dir() :
                newloc = (newscan1/fp.relative_to(oldscan1)).parent
                if not newloc.is_dir() :
                    newloc.mkdir(parents=True)
                yield fp,newloc
        oldxml = folder/'data'/slide_ID/'im3'/'xml'
        newxml = folder/'test_for_jenkins'/'segmentation'/'root'/slide_ID/'im3'/'xml'
        if not newxml.is_dir() :
            newxml.mkdir(parents=True)
        for fp in oldxml.glob('*') :
            yield fp,newxml

class TestSegmentationNNUNet(TestSegmentationBase) :
    """
    Class to use for testing the nnU-Net segmentation algorithm
    """

    @property
    def outputfilenames(self) :
        oldcomptiffs = folder/'data'/slide_ID/'inform_data'/'Component_Tiffs'
        outputdir = folder/'test_for_jenkins'/'segmentation'/'root'/slide_ID/'im3'/'segmentation'/'nnunet'
        all_fps = []
        for fns in [fp.name[:-len('_component_data.tif')] for fp in oldcomptiffs.glob('*_component_data.tif')] :
            all_fps.append(outputdir/f'{fns}_nnunet_nuclear_segmentation.npz')
        return all_fps

    def test_segmentation_nnunet(self) :
        #run the segmentation cohort with the nnU-Net algorithm
        args = [os.fspath(folder/'test_for_jenkins'/'segmentation'/'root'),
                slide_ID,
                '--njobs','5',
                '--allow-local-edits',
                '--selectrectangles'
                ]
        for rn in rectangle_ns_with_comp_tiff_files_nnunet :
            args.append(str(rn))
        SegmentationSampleNNUNet.runfromargumentparser(args=args)
        #compare the results to the reference files
        outputdir = folder/'test_for_jenkins'/'segmentation'/'root'/slide_ID/'im3'/'segmentation'/'nnunet'
        try :
            for fp in (folder/'data'/'reference'/'segmentation'/'nnunet').glob('*') :
                refa = (np.load(fp))['arr_0']
                testa = (np.load(outputdir/fp.name))['arr_0']
                np.testing.assert_allclose(testa,refa)
        except :
            self.saveoutput()
            raise
        finally :
            self.removeoutput()
            shutil.rmtree(folder/'test_for_jenkins'/'segmentation')

class TestSegmentationDeepCell(TestSegmentationBase) :
    """
    Class to use for testing the DeepCell segmentation algorithm
    """

    @property
    def outputfilenames(self) :
        oldcomptiffs = folder/'data'/slide_ID/'inform_data'/'Component_Tiffs'
        outputdir = folder/'test_for_jenkins'/'segmentation'/'root'/slide_ID/'im3'/'segmentation'/'deepcell'
        all_fps = []
        for fns in [fp.name[:-len('_component_data.tif')] for fp in oldcomptiffs.glob('*_component_data.tif')] :
            all_fps.append(outputdir/f'{fns}_deepcell_nuclear_segmentation.npz')
        return all_fps

    def test_segmentation_deepcell(self) :
        #run the segmentation cohort with the nnU-Net algorithm
        args = [os.fspath(folder/'test_for_jenkins'/'segmentation'/'root'),
                slide_ID,
                '--allow-local-edits',
                '--selectrectangles'
                ]
        for rn in rectangle_ns_with_comp_tiff_files_deepcell :
            args.append(str(rn))
        SegmentationSampleDeepCell.runfromargumentparser(args=args)
        #compare the results to the reference files
        outputdir = folder/'test_for_jenkins'/'segmentation'/'root'/slide_ID/'im3'/'segmentation'/'deepcell'
        try :
            for fp in (folder/'data'/'reference'/'segmentation'/'deepcell').glob('*') :
                refa = (np.load(fp))['arr_0']
                testa = (np.load(outputdir/fp.name))['arr_0']
                np.testing.assert_allclose(testa,refa)
        except :
            self.saveoutput()
            raise
        finally :
            self.removeoutput()
            shutil.rmtree(folder/'test_for_jenkins'/'segmentation')
