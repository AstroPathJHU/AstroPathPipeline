#imports
from .misc import cd
import numpy as np
import xml.etree.ElementTree as et
import os, glob, logging

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

#helper function to normalize a given image for exposure time layer-by-layer
def normalizeImageByExposureTime(raw_img,raw_fp,max_exp_times,metadata_top_dir) :
  if len(max_exp_times)!=raw_img.shape[-1] :
    raise RuntimeError(f"""ERROR: the list of max exposure times (length {len(max_exp_times)}) and the raw img ({raw_fp}) with shape 
                           {raw_img.shape} passed to normalizeImageByExposureTime don't match!""")
  nlayers = len(max_exp_times)
  raw_img_dtype = raw_img.dtype
  max_value = np.iinfo(raw_img_dtype).max
  exposure_times = getExposureTimesByLayer(raw_fp,nlayers,metadata_top_dir)
  normalized_img = raw_img.copy()
  for li in range(nlayers) :
    if exposure_times[li]!=max_exp_times[li] : #layer is only different if it isn't maximally exposed
      norm = exposure_times[li]/max_exp_times[li]
      norm_img_layer=(1.0*raw_img[:,:,li])/norm #converted to a float here
      n_clipped_pixels = np.sum(np.where(norm_img_layer>max_value,1,0))
      if n_clipped_pixels>0 : #warn the user if any pixels have been clipped as a result
        msg = f'WARNING: normalizing layer {li+1} of {raw_fp} by '
        msg+=f'{norm:.5f} = ({exposure_times[li]:.1f})/({max_exp_times[li]:.1f}) clips'
        msg+=f' {n_clipped_pixels} pixels'
        msg+=f' ({100.*n_clipped_pixels/(raw_img.shape[0]*raw_img.shape[1])}%)'
        msg+=f' with resulting flux > {max_value}'
        utility_logger.warn(msg)
      normalized_img[:,:,li] = (np.clip(np.rint(norm_img_layer),0,max_value)).astype(raw_img_dtype) #convert back to original datatype
  return normalized_img
