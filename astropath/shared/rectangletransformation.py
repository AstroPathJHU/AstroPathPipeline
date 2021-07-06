#imports
import abc
import numpy as np

class RectangleTransformationBase(abc.ABC):
  @abc.abstractmethod
  def transform(self, previousimage):
    """
    Takes in the previous image, returns the new image.
    """

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
      return (np.clip(np.rint(originalimage/self._flatfield),0,np.iinfo(raw_dtype).max)).astype(raw_dtype)
    else :
      return (originalimage/self._flatfield).astype(raw_dtype,casting='same_kind')