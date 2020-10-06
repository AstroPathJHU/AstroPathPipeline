#imports
from .img_file_io import getExposureTimesByLayer
import numpy as np
import cv2

#helper function to correct a single image layer for exposure time differences
def correctImageLayerForExposureTime(raw_img_layer,exp_time,med_exp_time,offset) :
  raw_img_dtype = raw_img_layer.dtype
  corr_img_layer = np.where((raw_img_layer-offset)>0,offset+(1.*med_exp_time/exp_time)*(raw_img_layer-offset),raw_img_layer) #converted to a float here
  if np.issubdtype(raw_img_dtype,np.integer) :
    max_value = np.iinfo(raw_img_dtype).max
    return (np.clip(np.rint(corr_img_layer),0,max_value)).astype(raw_img_dtype) #round, clip to range, and convert back to original datatype
  else :
    return (corr_img_layer).astype(raw_img_dtype) #otherwise just convert back to original datatype

#helper function to normalize a given image for exposure time layer-by-layer
def correctImageForExposureTime(raw_img,raw_fp,metadata_top_dir,med_exp_times,correction_offsets) :
  if len(raw_img.shape)!=3 :
    raise RuntimeError(f"""ERROR: correctImageForExposureTime only runs on multilayer images but was called on an image with shape {raw_img.shape}.
                             Use correctImageLayerForExposureTime instead.""")
  if len(med_exp_times)!=raw_img.shape[-1] :
    raise RuntimeError(f"""ERROR: the list of median exposure times (length {len(med_exp_times)}) and the raw img ({raw_fp}) with shape 
                           {raw_img.shape} passed to correctImageForExposureTime don't match!""")
  if len(correction_offsets)!=raw_img.shape[-1] :
    raise RuntimeError(f"""ERROR: the list of correction offsets (length {len(correction_offsets)}) and the raw img ({raw_fp}) with shape 
                           {raw_img.shape} passed to correctImageForExposureTime don't match!""")
  nlayers = raw_img.shape[-1]
  exposure_times = getExposureTimesByLayer(raw_fp,nlayers,metadata_top_dir)
  corrected_img = raw_img.copy()
  for li in range(nlayers) :
    if exposure_times[li]!=med_exp_times[li] : #layer is only different if it isn't already at the median exposure time
      corrected_img[:,:,li] = correctImageLayerForExposureTime(raw_img[:,:,li],exposure_times[li],med_exp_times[li],correction_offsets[li])
  return corrected_img

#helper function to apply a flatfield to a given image layer
def correctImageLayerWithFlatfield(raw_img_layer,flatfield_layer) :
  raw_dtype = raw_img_layer.dtype
  if np.issubdtype(raw_dtype,np.integer) :
    ff_corrected_layer = (np.clip(np.rint(raw_img_layer/flatfield_layer),0,np.iinfo(raw_dtype).max)).astype(raw_dtype)
  else :
    ff_corrected_layer = (raw_img_layer/flatfield_layer).astype(raw_dtype)
  return ff_corrected_layer

#helper function to apply a multilayer flatfield to a multilayer image
def correctImageWithFlatfield(raw_img,flatfield) :
  raw_dtype = raw_img.dtype
  if np.issubdtype(raw_dtype,np.integer) :
    ff_corrected = (np.clip(np.rint(raw_img/flatfield),0,np.iinfo(raw_dtype).max)).astype(raw_dtype)
  else :
    ff_corrected = (raw_img/flatfield).astype(raw_dtype)
  return ff_corrected

#helper function to correct an image layer with given warp dx and dy fields
def correctImageLayerWithWarpFields(raw_img_layer,dx_warps,dy_warps) :
  grid = np.mgrid[0:raw_img_layer.shape[0],0:raw_img_layer.shape[1]]
  xpos, ypos = grid[1], grid[0]
  map_x = (xpos-dx_warps).astype(np.float32) 
  map_y = (ypos-dy_warps).astype(np.float32)
  return cv2.remap(raw_img_layer,map_x,map_y,cv2.INTER_LINEAR)
