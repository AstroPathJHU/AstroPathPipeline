#imports
import methodtools
import numpy as np
from ...shared.sample import SampleWithSegmentationFolder, WorkflowSample, ParallelSample 
from ...shared.sample import ReadRectanglesComponentTiffFromXML, ReadRectanglesComponentAndIHCTiffFromXML
from .config import SEG_CONST

class SegmentationSampleBase(SampleWithSegmentationFolder,WorkflowSample,ParallelSample) :
    """
    Base class for all segmentation samples in general regardless of the images on which they're meant to run
    """

    def run(self,**kwargs) :
        if not self.segmentationfolder.is_dir() :
            self.segmentationfolder.mkdir(parents=True)
        self.runsegmentation(**kwargs)

    def inputfiles(self, **kwargs):
        return super().inputfiles(**kwargs)

    @methodtools.lru_cache()
    def runsegmentation(self, **kwargs): 
        pass

    @classmethod
    def segmentationalgorithm(cls):
        return "none"

    @classmethod
    def workflowdependencyclasses(cls, **kwargs):
        return super().workflowdependencyclasses(**kwargs)

    @classmethod
    def getoutputfiles(cls,*args,**kwargs) :
        return []

    @classmethod
    def getsegfileappend(cls) :
        append = None
        if cls.segmentationalgorithm()=='nnunet' :
            append = SEG_CONST.NNUNET_SEGMENT_FILE_APPEND
        elif cls.segmentationalgorithm()=='deepcell' :
            append = SEG_CONST.DEEPCELL_SEGMENT_FILE_APPEND
        elif cls.segmentationalgorithm()=='mesmer' :
            append = SEG_CONST.MESMER_SEGMENT_FILE_APPEND
        else :
            raise ValueError(f'ERROR: unrecognized segmentation algorithm "{cls.segmentationalgorithm}"')
        return append

class SegmentationSampleUsingComponentTiff(SegmentationSampleBase,ReadRectanglesComponentTiffFromXML) :
    """
    Base class for segmentation samples that read at least one layer from the component tiff images
    """

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.componenttifffile for r in self.rectangles),
               ]

    @classmethod
    def getoutputfiles(cls,SlideID,segmentationroot,informdataroot,segmentationfolderarg,**otherworkflowkwargs) :
        outputdir=cls.segmentation_folder(segmentationfolderarg,segmentationroot,SlideID)
        append = cls.getsegfileappend()
        file_stems = [fp.name[:-len('_component_data.tif')] 
                      for fp in (informdataroot/SlideID/'inform_data'/'Component_Tiffs').glob('*_component_data.tif')]
        outputfiles = []
        for stem in file_stems :
            outputfiles.append(outputdir/f'{stem}_{append}')
        return outputfiles

class SegmentationSampleDAPIComponentTiff(SegmentationSampleUsingComponentTiff) :
    """
    Base class for segmentation samples that need only the DAPI layer of the component tiff images
    """

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,layercomponenttiff='setlater',layerscomponenttiff='setlater',**kwargs)
        dapi_layer_n = None
        for layer,opal,_ in self.layers_opals_targets :
            if opal=='dapi' :
                dapi_layer_n = layer
                break
        if dapi_layer_n is None :
            errmsg =  'ERROR: could not determine which layer of the component tiff is the DAPI layer! '
            errmsg+= f'layers/opals/targets = {self.layers_opals_targets}'
            raise RuntimeError(errmsg)
        self.setlayerscomponenttiff(layercomponenttiff=dapi_layer_n)

class SegmentationSampleDAPIComponentMembraneIHCTiff(SegmentationSampleDAPIComponentTiff,
                                                     ReadRectanglesComponentAndIHCTiffFromXML) :
    """
    Base class for segmentation samples that need the DAPI layer from the component tiff and
    a membrane layer from the deconvolved IHC image
    """

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.ihctifffile for r in self.rectangles),
               ]

    def get_membrane_layer_from_ihc_image(self,ihc_image) :
        pca_vec_to_dot = np.expand_dims(SEG_CONST.IHC_PCA_BLACK_COMPONENT,0).T
        im_membrane = -1.0*(np.dot(ihc_image,pca_vec_to_dot))[:,:,0]
        membrane_layer = (im_membrane-np.min(im_membrane))/SEG_CONST.IHC_MEMBRANE_LAYER_NORM
        return membrane_layer

class SegmentationSampleDAPIMembraneComponentTiff(SegmentationSampleUsingComponentTiff) :
    """
    Base class for segmentation samples that need both a DAPI and a membrane layer from the component tiff
    """

    multilayercomponenttiff = True    
    
    def __init__(self,*args,**kwargs) :
        super().__init__(*args,layercomponenttiff='setlater',layerscomponenttiff='setlater',**kwargs)
        dapi_layer_n = None; membrane_layer_n = None
        for layer,opal,target in self.layers_opals_targets :
            if opal=='dapi' :
                dapi_layer_n = layer
            elif target in SEG_CONST.MEMBRANE_LAYER_TARGETS :
                membrane_layer_n = layer
        if dapi_layer_n is None :
            errmsg =  'ERROR: could not determine which layer of the component tiff is the DAPI layer! '
            errmsg+= f'layers/opals/targets = {self.layers_opals_targets}'
            raise RuntimeError(errmsg)
        if membrane_layer_n is None :
            errmsg =  'ERROR: could not determine which layer of the component tiff is the membrane layer! '
            errmsg+= f'layers/opals/targets = {self.layers_opals_targets}'
            raise RuntimeError(errmsg)
        self.setlayerscomponenttiff(layerscomponenttiff=[dapi_layer_n,membrane_layer_n])
