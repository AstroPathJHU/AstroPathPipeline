#imports
from .misc import cd
import numpy as np
import xml.etree.ElementTree as et
import os, glob, cv2, logging

#global variables
RAWFILE_EXT           = '.Data.dat'
PARAMETER_XMLFILE_EXT = '.Parameters.xml'
EXPOSURE_XML_EXT      = '.SpectralBasisInfo.Exposure.xml'
#logger
utility_logger = logging.getLogger("utility")
utility_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s    [%(funcName)s, %(asctime)s]"))
utility_logger.addHandler(handler)

#helper function to read the binary dump of a raw im3 file 
def im3readraw(f,dtype=np.uint16) :
  with open(f,mode='rb') as fp : #read as binary
    content = np.memmap(fp,dtype=dtype,mode='r')
  return content

#helper function to write an array of uint16s as an im3 file
def im3writeraw(outname,a) :
  with open(outname,mode='wb') as fp : #write as binary
    a.tofile(fp)

#helper function to read a raw image file and return it as an array of shape (height,width,n_layers)
def getRawAsHWL(fname,height,width,nlayers,dtype=np.uint16) :
  #get the .raw file as a vector of uint16s
  img = im3readraw(fname,dtype)
  #reshape it to the given dimensions
  try :
    img_a = np.reshape(img,(nlayers,width,height),order="F")
  except ValueError :
    msg = f"ERROR: Raw image file shape ({nlayers} layers, {len(img)} total bytes) is mismatched to"
    msg+= f" dimensions (layers, width={width}, height={height})!"
    raise ValueError(msg)
  #flip x and y dimensions to display image correctly, move layers to z-axis
  img_to_return = np.transpose(img_a,(2,1,0))
  return img_to_return

#helper function to read a single-layer image and return it as an array of shape (height,width)
def getRawAsHW(fname,height,width,dtype=np.uint16) :
  #get the file as a vector of uint16s
  img = im3readraw(fname,dtype)
  #reshape it
  try :
    img_a = np.reshape(img,(height,width),order="F")
  except ValueError :
    msg = f"ERROR: single layer image file ({len(img)} total bytes) shape is mismatched to"
    msg+= f" dimensions (width={width}, height={height})!"
    raise ValueError(msg)
  return img_a

#helper function to flatten and write out a given image as binary uint16 content
def writeImageToFile(img_array,filename_to_write,dtype=np.uint16) :
  #write out image flattened in fortran order
  im3writeraw(filename_to_write,img_array.flatten(order="F").astype(dtype))

#helper function to smooth an image
#this can be run in parallel
def smoothImageWorker(im_array,smoothsigma,return_list=None) :
  if return_list is not None :
    im_in_umat = cv2.UMat(im_array)
    im_out_umat = cv2.UMat(np.empty_like(im_array))
    cv2.GaussianBlur(im_in_umat,(0,0),smoothsigma,im_out_umat,borderType=cv2.BORDER_REPLICATE)
    return_list.append(im_out_umat.get())
  else :
    return cv2.GaussianBlur(im_array,(0,0),smoothsigma,borderType=cv2.BORDER_REPLICATE)

#helper function to get an image dimension tuple from the sample XML file
def getImageHWLFromXMLFile(metadata_topdir,samplename) :
    xmlfile_path = os.path.join(metadata_topdir,samplename,'im3','xml',f'{samplename}{PARAMETER_XMLFILE_EXT}')
    tree = et.parse(xmlfile_path)
    for child in tree.getroot() :
        if child.attrib['name']=='Shape' :
            img_width, img_height, img_nlayers = tuple([int(val) for val in (child.text).split()])
    return img_height, img_width, img_nlayers

#helper function to get a list of exposure times by each layer for a given raw image
#fp can be a path to a raw file or to an exposure XML file 
#but if it's a raw file the metadata top dir must also be provided
def getExposureTimesByLayer(fp,nlayers,metadata_top_dir=None) :
  layer_exposure_times_to_return = []
  if RAWFILE_EXT in fp :
    if metadata_top_dir is None :
      raise RuntimeError(f'ERROR: metadata top dir must be supplied to get exposure times fo raw file path {fp}!')
    sample_name = os.path.basename(os.path.dirname(os.path.normpath(fp)))
    xmlfile_path = os.path.join(metadata_top_dir,sample_name,'im3','xml',os.path.basename(os.path.normpath(fp)).replace(RAWFILE_EXT,EXPOSURE_XML_EXT))
  elif EXPOSURE_XML_EXT in fp :
    xmlfile_path = fp
  else :
    raise ValueError(f"ERROR: file path {fp} given to getExposureTimesByLayer doesn't represent a recognized raw or exposure xml file!")
  if not os.path.isfile(xmlfile_path) :
    raise RuntimeError(f"ERROR: {xmlfile_path} searched in getExposureTimesByLayer not found!")
  root = (et.parse(xmlfile_path)).getroot()
  for li in range(nlayers) :
    if nlayers==35 :
      if li in range(9) :
        thislayer_exposure_time=float(((root[0].text).split())[li])
      elif li in range(9,18) :
        thislayer_exposure_time=float(((root[1].text).split())[li-9])
      elif li in range(18,25) :
        thislayer_exposure_time=float(((root[2].text).split())[li-18])
      elif li in range(25,32) :
        thislayer_exposure_time=float(((root[3].text).split())[li-25])
      elif li in range(32,35) :
        thislayer_exposure_time=float(((root[4].text).split())[li-32])
    elif nlayers==43 :
      if li in range(9) :
        thislayer_exposure_time=float(((root[0].text).split())[li])
      elif li in range(9,11) :
        thislayer_exposure_time=float(((root[1].text).split())[li-9])
      elif li in range(11,17) :
        thislayer_exposure_time=float(((root[2].text).split())[li-11])
      elif li in range(17,20) :
        thislayer_exposure_time=float(((root[3].text).split())[li-17])
      elif li in range(20,29) :
        thislayer_exposure_time=float(((root[4].text).split())[li-20])
      elif li in range(29,36) :
        thislayer_exposure_time=float(((root[5].text).split())[li-29])
      elif li in range(36,43) :
        thislayer_exposure_time=float(((root[6].text).split())[li-36])
    else :
      raise ValueError(f"ERROR: number of image layers ({nlayers}) passed to getExposureTimesByLayer is not a recognized option!")
    layer_exposure_times_to_return.append(thislayer_exposure_time)
  return layer_exposure_times_to_return

#helper function to return a list of the maximum exposure times observed in each layer of a given sample
def getSampleMaxExposureTimesByLayer(metadata_topdir,samplename) :
    _,_,nlayers = getImageHWLFromXMLFile(metadata_topdir,samplename)
    max_exposure_times_by_layer = []
    for li in range(nlayers) :
        max_exposure_times_by_layer.append(0)
    with cd(os.path.join(metadata_topdir,samplename,'im3','xml')) :
        all_fps = [os.path.join(metadata_topdir,samplename,'im3','xml',fn) for fn in glob.glob(f'*{EXPOSURE_XML_EXT}')]
    utility_logger.info(f'Finding maximum exposure times in a sample of {len(all_fps)} images with {nlayers} layers each....')
    for fp in all_fps :
        this_image_layer_exposure_times = getExposureTimesByLayer(fp,nlayers)
        for li in range(nlayers) :
            if this_image_layer_exposure_times[li]>max_exposure_times_by_layer[li] :
                max_exposure_times_by_layer[li] = this_image_layer_exposure_times[li]
    return max_exposure_times_by_layer

#helper function to correct a single image layer for exposure time differences
def correctImageLayerForExposureTime(raw_img_layer,exp_time,max_exp_time,offset) :
  raw_img_dtype = raw_img_layer.dtype
  max_value = np.iinfo(raw_img_dtype).max
  corr_img_layer = np.where((raw_img_layer-offset)>0,offset+(1.*max_exp_time/exp_time)*(raw_img_layer-offset),raw_img_layer) #converted to a float here
  return (np.clip(np.rint(corr_img_layer),0,max_value)).astype(raw_img_dtype) #round, clip to range, and convert back to original datatype

#helper function to normalize a given image for exposure time layer-by-layer
def correctImageForExposureTime(raw_img,raw_fp,metadata_top_dir,max_exp_times,correction_offsets) :
  if len(raw_img.shape)!=3 :
    raise RuntimeError(f"""ERROR: correctImageForExposureTime only runs on multilayer images but was called on an image with shape {raw_img.shape}.
                             Use correctImageLayerForExposureTime instead.""")
  if len(max_exp_times)!=raw_img.shape[-1] :
    raise RuntimeError(f"""ERROR: the list of max exposure times (length {len(max_exp_times)}) and the raw img ({raw_fp}) with shape 
                           {raw_img.shape} passed to correctImageForExposureTime don't match!""")
  if len(correction_offsets)!=raw_img.shape[-1] :
    raise RuntimeError(f"""ERROR: the list of correction offsets (length {len(correction_offsets)}) and the raw img ({raw_fp}) with shape 
                           {raw_img.shape} passed to correctImageForExposureTime don't match!""")
  nlayers = raw_img.shape[-1]
  exposure_times = getExposureTimesByLayer(raw_fp,nlayers,metadata_top_dir)
  corrected_img = raw_img.copy()
  for li in range(nlayers) :
    if exposure_times[li]!=max_exp_times[li] : #layer is only different if it isn't maximally exposed
      corrected_img[:,:,li] = correctImageLayerForExposureTime(raw_img[:,:,li],exposure_times[li],max_exp_times[li],correction_offsets[li])
  return corrected_img
