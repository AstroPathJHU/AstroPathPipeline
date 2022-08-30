#imports
import abc, cv2
import numpy as np
from numba import njit

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

  def __init__(self, ets, offsets) :
    self._exp_times = (np.array(ets))[np.newaxis,np.newaxis,:]
    self._offsets = (np.array(offsets))[np.newaxis,np.newaxis,:]
    self._med_ets = None
    self._nlayers = self._exp_times.shape[-1]

  def set_med_ets(self, med_ets):
    self._med_ets = (med_ets)[np.newaxis,np.newaxis,:]
    if not (self._exp_times.shape[-1]==self._med_ets.shape[-1]==self._offsets.shape[-1]) :
      errmsg = f'ERROR: exposure times (length {self._exp_times.shape[-1]}), median exposure times '
      errmsg+= f'(length {self._med_ets.shape[-1]}), and dark current offsets (length {self._offsets.shape[-1]}) '
      errmsg+= 'must all be the same length!'
      raise ValueError(errmsg)
  
  def transform(self, originalimage) :
    if self._med_ets is None:
      raise ValueError("Have to call set_med_ets before transforming")
    if self._nlayers!=originalimage.shape[-1] :
      errmsg = f'ERROR: image with shape {originalimage.shape} cannot be corrected for exposure time '
      errmsg+= f'using a setup for images with {self._nlayers} layers!'
      raise ValueError(errmsg)
    raw_img_dtype = originalimage.dtype
    return self.__get_corrected_image(originalimage,
                                      self._exp_times,self._med_ets,self._offsets,
                                      raw_img_dtype,np.issubdtype(raw_img_dtype,np.integer)) 

  @staticmethod
  @njit
  def __get_corrected_image(originalimage,exp_times,med_ets,offsets,original_dtype,original_is_int) :
    corr_img = np.where((originalimage-offsets)>0,
                        offsets+(1.*med_ets/exp_times)*(originalimage-offsets),
                        originalimage) #converted to a float here
    if original_is_int :
      #round, clip to range, and convert back to original datatype
      max_value = np.iinfo(original_dtype).max
      return (np.clip(np.rint(corr_img),0,max_value)).astype(original_dtype) 
    else :
      return corr_img.astype(original_dtype) #otherwise just convert back to original datatype

class RectangleExposureTimeTransformationSingleLayer(RectangleTransformationBase):
  """
  Corrects a single layer of a rectangle image's illumination given:

  et :     exposure time in the image layer
  med_et : the median exposure time in the layer over all sample rectangles 
           (or any other exposure time to correct the image layer to) 
  offset : the dark current offset in the layer

  Any images given to the transform function must only have one layer!
  """

  def __init__(self, et, offset) :
    self._exp_time = et
    self._med_et = None
    self._offset = offset
  
  def set_med_et(self, med_et):
    self._med_et = med_et
  
  def transform(self, originalimage) :
    if self._med_et is None:
      raise ValueError("Have to call set_med_et before transforming")
    if (len(originalimage.shape)==3 and originalimage.shape[2]!=1) or len(originalimage.shape)>3 :
      errmsg = f'ERROR: image layer with shape {originalimage.shape} cannot be corrected for exposure time '
      errmsg+= 'using a setup for single layer images!'
      raise ValueError(errmsg)
    raw_img_dtype = originalimage.dtype
    return self.__get_corrected_image_layer(originalimage,
                                            self._exp_time,self._med_et,self._offset,
                                            raw_img_dtype,np.issubdtype(raw_img_dtype,np.integer))

  @staticmethod
  @njit
  def __get_corrected_image_layer(originalimage,exp_time,med_et,offset,original_dtype,original_is_int) :
    corr_img = np.where((originalimage-offset)>0,
                        offset+(1.*med_et/exp_time)*(originalimage-offset),
                        originalimage) #converted to a float here
    if original_is_int :
      #round, clip to range, and convert back to original datatype
      max_value = np.iinfo(original_dtype).max
      return (np.clip(np.rint(corr_img),0,max_value)).astype(original_dtype) 
    else :
      return (corr_img).astype(original_dtype) #otherwise just convert back to original datatype

class RectangleFlatfieldTransformationMultilayer(RectangleTransformationBase):
  """
  Divides a rectangle image by given flatfield correction factors
  flatfield: flatfield correction factors (must be same dimensions as images to correct)
  """

  def __init__(self, flatfield=None) :
    self._flatfield = flatfield

  def set_flatfield(self, flatfield):
    self._flatfield = flatfield

  def transform(self, originalimage) :
    if self._flatfield is None:
      raise ValueError("Have to set the flatfield before transforming")
    if not self._flatfield.shape==originalimage.shape :
      errmsg = f'ERROR: shape mismatch (flatfield.shape = {self._flatfield.shape}, originalimage.shape '
      errmsg+= f'= {originalimage.shape}) in RectangleFlatfieldTransformationMultilayer.transform!'
      raise ValueError(errmsg)
    raw_dtype = originalimage.dtype
    return self.__get_corrected_image(originalimage,self._flatfield,raw_dtype,np.issubdtype(raw_dtype,np.integer))

  @staticmethod
  @njit
  def __get_corrected_image(originalimage,flatfield,original_dtype,original_is_int) :
    if original_is_int :
      return (np.clip(np.rint(originalimage/flatfield),0,np.iinfo(original_dtype).max)).astype(original_dtype)
    else :
      return (originalimage/flatfield).astype(original_dtype)

class RectangleFlatfieldTransformationSinglelayer(RectangleTransformationBase):
  """
  Divides a rectangle image layer by given flatfield correction factors
  flatfield_layer: flatfield correction factors
  """

  def __init__(self, flatfield_layer=None) :
    self._flatfield = flatfield_layer

  def set_flatfield(self, flatfield):
    self._flatfield = flatfield
  
  def transform(self, originalimage) :
    if self._flatfield is None:
      raise ValueError("Have to set the flatfield before transforming")
    if ( (len(originalimage.shape)==3 and (originalimage.shape[2]!=1 or self._flatfield.shape!=originalimage.shape[:-1])) or 
         (len(originalimage.shape)==2 and self._flatfield.shape!=originalimage.shape) or (len(originalimage.shape)>3) ) :
      errmsg = f'ERROR: shape mismatch (flatfield.shape = {self._flatfield.shape}, originalimage.shape '
      errmsg+= f'= {originalimage.shape}) in RectangleFlatfieldTransformationSinglelayer.transform!'
      raise ValueError(errmsg)
    raw_dtype = originalimage.dtype
    if len(originalimage.shape)==3 :
      flatfield = self._flatfield[:,:,np.newaxis]
    else :
      flatfield = self._flatfield
    return self.__get_corrected_image_layer(originalimage,flatfield,raw_dtype,np.issubdtype(raw_dtype,np.integer))

  @staticmethod
  @njit
  def __get_corrected_image_layer(originalimage,flatfield,original_dtype,original_is_int) :
    if original_is_int :
      return (np.clip(np.rint(originalimage/flatfield),0,np.iinfo(original_dtype).max)).astype(original_dtype)
    else :
      return (originalimage/flatfield).astype(original_dtype)

class RectangleWarpingTransformationMultilayer(RectangleTransformationBase) :
  """
  Applies a set of defined warping objects to an image and returns the result
  """

  def __init__(self,warps_by_layer=None) :
    self._warps_by_layer = warps_by_layer

  def set_warp(self, warp):
    self._warps_by_layer = warp
  
  def transform(self, originalimage) :
    if self._warps_by_layer is None:
      raise ValueError("Have to set the warp before transforming")
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

  def __init__(self,warp=None) :
    self._warp = warp

  def set_warp(self, warp):
    self._warp = warp
  
  def transform(self, originalimage) :
    if self._warp is None:
      raise ValueError("Have to set the warp before transforming")
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

class ImageTransformation(RectangleTransformationBase):
  def __init__(self, transformation):
    self.__transformation = transformation
  def transform(self, originalimage):
    return self.__transformation(originalimage)

class AsTypeTransformation(RectangleTransformationBase):
  def __init__(self, dtype):
    self.__dtype = dtype
  def transform(self, originalimage):
    return originalimage.astype(self.__dtype)
