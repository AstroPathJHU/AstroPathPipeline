#imports
import abc, cv2
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

  def __init__(self, ets, med_ets, offsets) :
    self._exp_times = np.array(ets)
    self._med_ets = np.array(med_ets)
    self._offsets = np.array(offsets)
    if not (len(self._exp_times)==len(self._med_ets)==len(self._offsets)) :
      errmsg = f'ERROR: exposure times (length {len(self._exp_times)}), median exposure times (length {len(self._med_ets)}),'
      errmsg+= f' and dark current offsets (length {len(self._offsets)}) must all be the same length!'
      raise ValueError(errmsg)
    self._nlayers = len(self._exp_times)
  
  def transform(self, originalimage) :
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

class RectangleExposureTimeTransformationSingleLayer(RectangleTransformationBase):
  """
  Corrects a single layer of a rectangle image's illumination given:

  et :     exposure time in the image layer
  med_et : the median exposure time in the layer over all sample rectangles 
           (or any other exposure time to correct the image layer to) 
  offset : the dark current offset in the layer

  Any images given to the transform function must only have one layer!
  """

  def __init__(self, et, med_et, offset) :
    self._exp_time = et
    self._med_et = med_et
    self._offset = offset
  
  def transform(self, originalimage) :
    if (len(originalimage.shape)==3 and originalimage.shape[2]!=1) or len(originalimage.shape)>3 :
      errmsg = f'ERROR: image layer with shape {originalimage.shape} cannot be corrected for exposure time '
      errmsg+= 'using a setup for single layer images!'
      raise ValueError(errmsg)
    raw_img_dtype = originalimage.dtype
    corr_img = np.where((originalimage-self._offset)>0,
                        self._offset+(1.*self._med_et/self._exp_time)*(originalimage-self._offset),
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

  def __init__(self, flatfield) :
    self._flatfield = flatfield
  
  def transform(self, originalimage) :
    if not self._flatfield.shape==originalimage.shape :
      errmsg = f'ERROR: shape mismatch (flatfield.shape = {self._flatfield.shape}, originalimage.shape '
      errmsg+= f'= {originalimage.shape}) in RectangleFlatfieldTransformationMultilayer.transform!'
      raise ValueError(errmsg)
    raw_dtype = originalimage.dtype
    if np.issubdtype(raw_dtype,np.integer) :
      return (np.clip(np.rint(originalimage/self._flatfield),0,np.iinfo(raw_dtype).max)).astype(raw_dtype)
    else :
      return (originalimage/self._flatfield).astype(raw_dtype,casting='same_kind')

class RectangleFlatfieldTransformationSinglelayer(RectangleTransformationBase):
  """
  Divides a rectangle image layer by given flatfield correction factors
  flatfield_layer: flatfield correction factors
  """

  def __init__(self, flatfield_layer) :
    self._flatfield = flatfield_layer
  
  def transform(self, originalimage) :
    if ( (len(originalimage.shape)==3 and (originalimage.shape[2]!=1 or self._flatfield.shape!=originalimage.shape[:-1])) or 
         (len(originalimage.shape)==2 and self._flatfield.shape!=originalimage.shape) or (len(originalimage.shape)>3) ) :
      errmsg = f'ERROR: shape mismatch (flatfield.shape = {self._flatfield.shape}, originalimage.shape '
      errmsg+= f'= {originalimage.shape}) in RectangleFlatfieldTransformationSinglelayer.transform!'
      raise ValueError(errmsg)
    raw_dtype = originalimage.dtype
    if np.issubdtype(raw_dtype,np.integer) :
      if len(originalimage.shape)==3 :
        return (np.clip(np.rint(originalimage/self._flatfield[:,:,np.newaxis]),0,np.iinfo(raw_dtype).max)).astype(raw_dtype)
      else :
        return (np.clip(np.rint(originalimage/self._flatfield),0,np.iinfo(raw_dtype).max)).astype(raw_dtype)
    else :
      if len(originalimage.shape)==3 :
        return (originalimage/self._flatfield[:,:,np.newaxis]).astype(raw_dtype,casting='same_kind')
      else :
        return (originalimage/self._flatfield).astype(raw_dtype,casting='same_kind')

class RectangleWarpingTransformationMultilayer(RectangleTransformationBase) :
  """
  Applies a set of defined warping objects to an image and returns the result
  """

  def __init__(self,warps_by_layer) :
    self._warps_by_layer = warps_by_layer

  def transform(self, originalimage) :
    corr_img = np.empty_like(originalimage)
    for li in range(originalimage.shape[-1]) :
      if self._warps_by_layer[li] is None :
        corr_img[:,:,li] = originalimage[:,:,li]
      else :
        layer_in_umat = cv2.UMat(originalimage[:,:,li])
        layer_out_umat = cv2.UMat(corr_img[:,:,li])
        self._warps_by_layer[li].warpLayerInPlace(layer_in_umat,layer_out_umat)
        corr_img[:,:,li] = layer_out_umat.get()
    return corr_img

class RectangleWarpingTransformationSinglelayer(RectangleTransformationBase) :
  """
  Applies a warping object to an image layer and returns the result
  """

  def __init__(self,warp) :
    self._warp = warp

  def transform(self, originalimage) :
    corr_img = np.empty_like(originalimage)
    if originalimage.shape[0]!=self._warp.m or originalimage.shape[1]!=self._warp.n :
      errmsg = f'ERROR: RectangleWarpingTransformationSinglelayer was passed an image with shape {originalimage.shape} '
      errmsg+= f'but a warp definition with (m,n) = ({self._warp.m},{self._warp.n})'
      raise ValueError(errmsg)
    layer_in_umat = cv2.UMat(originalimage)
    layer_out_umat = cv2.UMat(corr_img)
    self._warp.warpLayerInPlace(layer_in_umat,layer_out_umat)
    corr_img = layer_out_umat.get()
    return corr_img
