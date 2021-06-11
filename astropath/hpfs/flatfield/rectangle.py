#imports
from ...shared.rectangle import RectangleReadIm3MultiLayer, RectangleTransformationBase
from ...utilities.img_correction import correctImageLayerForExposureTime, correctImageLayerWithFlatfield
from ...utilities.img_file_io import getExposureTimesByLayer
from ...utilities.config import CONST as UNIV_CONST
import pathlib, numpy as np

class RectangleExposureTimeTransformationMultiLayer(RectangleTransformationBase):
    """
    Corrects an entire rectangle image's illumination given lists of:

    ets:     exposure times by layer 
    med_ets: the median exposure times in each layer over all sample rectangles 
             (or any other exposure times to correct the image to) 
    offsets: the dark current offsets by layer
    
    The three lists must have lengths equal to the number of image layers.
    """

    def __init__(self, ets, med_ets, offsets):
        self._exp_times = np.array(ets)
        self._med_ets = np.array(med_ets)
        self._offsets = np.array(offsets)
        if not (len(self._exp_times)==len(self._med_ets)==len(self._offsets)) :
            errmsg = f'ERROR: exposure times (length {len(self._exp_times)}), median exposure times (length {len(self._med_ets)}),'
            errmsg+= f' and dark current offsets (length {len(self._offsets)}) must all be the same length!'
            raise ValueError(errmsg)
        self._nlayers = len(self._exp_times)
    
    def transform(self, originalimage):
        if self._nlayers!=originalimage.shape[-1] :
            errmsg = f'ERROR: image with shape {originalimage.shape} cannot be corrected for exposure time '
            errmsg+= f'using a setup for images with {self._nlayers} layers!'
            raise ValueError(errmsg)
        raw_img_dtype = originalimage.dtype
        exp_times = self._exp_times[np.newaxis,np.newaxis,:]
        med_ets = self._med_ets[np.newaxis,np.newaxis,:]
        offsets = self._offsets[np.newaxis,np.newaxis,:]
        corr_img = np.where((originalimage-offsets)>0,
                            offsets+(1.*med_ets/exp_times)*(originalimage-offsets),
                            originalimage) #converted to a float here
        if np.issubdtype(raw_img_dtype,np.integer) :
            #round, clip to range, and convert back to original datatype
            max_value = np.iinfo(raw_img_dtype).max
            return (np.clip(np.rint(corr_img),0,max_value)).astype(raw_img_dtype) 
        else :
            return (corr_img).astype(raw_img_dtype,casting='same_kind') #otherwise just convert back to original datatype

class RectangleFlatfieldTransformationMultilayer(RectangleTransformationBase):
    """
    Divides a rectangle image by given flatfield correction factors
    flatfield: flatfield correction factors (must be same dimensions as images to correct)
    """

    def __init__(self, flatfield):
        self._flatfield = flatfield
    
    def transform(self, originalimage):
        if not self._flatfield.shape==originalimage.shape :
            errmsg = f'ERROR: shape mismatch (flatfield.shape = {self._flatfield.shape}, originalimage.shape '
            errmsg+= f'= {originalimage.shape}) in RectangleFlatfieldTransformationMultilayer.transform!'
            raise ValueError(errmsg)
        raw_dtype = originalimage.dtype
        if np.issubdtype(raw_dtype,np.integer) :
            return (np.clip(np.rint(originalimage/flatfield),0,np.iinfo(raw_dtype).max)).astype(raw_dtype)
        else :
            return (originalimage/flatfield).astype(raw_dtype,casting='same_kind')

class RectangleCorrectedIm3MultiLayer(RectangleReadIm3MultiLayer):
    """
    Class for Rectangles whose multilayer im3 data should be corrected for differences in exposure time 
    and/or flatfielding (either or both can be omitted)
    """
    _DEBUG = False #we're going to be running some multithreaded processes on these and so we likely will need to get the images multiple times
    
    def __post_init__(self, *args, transformations=None, **kwargs) :
        if transformations is None : 
            transformations = []
        super().__post_init__(*args, transformations=transformations, **kwargs)

    def add_exposure_time_correction_transformation(self,med_ets,offsets) :
        """
        Add a transformation to a rectangle to correct it for differences in exposure time given:

        med_ets = the median exposure times in the rectangle's slide 
        offsets = the list of dark current offsets for the rectangle's slide
        """
        if (med_ets is not None) and (offsets is not None) :
            self.add_transformation(RectangleExposureTimeTransformationMultiLayer(self.allexposuretimes,med_ets,offsets))

    def add_flatfield_correction_transformation(self,flatfield) :
        """
        Add a transformation to a rectangle to correct it with a given flatfield

        flatfield = the flatfield correction factor image to apply
        """
        if flatfield is not None:
            self.add_transformation(RectangleFlatfieldTransformationMultilayer(flatfield))
