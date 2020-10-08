#imports
from .tableio import readtable
from .misc import cd
import numpy as np
import xml.etree.ElementTree as et
import os, glob, cv2, logging, dataclasses

#global variables
PARAMETER_XMLFILE_EXT = '.Parameters.xml'
EXPOSURE_XML_EXT      = '.SpectralBasisInfo.Exposure.xml'
#logger
utility_logger = logging.getLogger("utility")
utility_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s    [%(funcName)s, %(asctime)s]"))
utility_logger.addHandler(handler)

#helper class to store exposure time offset factor for a single layer (with some extra info)
@dataclasses.dataclass
class LayerOffset :
  layer_n    : int
  n_overlaps : int
  offset     : float
  final_cost : float

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

#helper function to flatten and write out a given image as binary content
def writeImageToFile(img_array,filename_to_write,dtype=np.uint16) :
  #if the image is three dimensional (with the smallest dimension last, probably the number of layers), it has to be transposed
  if len(img_array.shape)==3 and (img_array.shape[2]<img_array.shape[0] and img_array.shape[2]<img_array.shape[1]) :
    img_array = img_array.transpose(2,1,0)
  elif len(img_array.shape)!=2 :
    msg = f'ERROR: writeImageToFile was passed an image of shape {img_array.shape}'
    msg+= ' instead of a 2D image, or a 3D multiplexed image with the number of layers last.'
    msg+= ' This might cause problems in writing it out in the right shape!'
    raise ValueError(msg)
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
  subdir_filepath = os.path.join(metadata_topdir,samplename,'im3','xml',f'{samplename}{PARAMETER_XMLFILE_EXT}')
  if os.path.isfile(subdir_filepath) :
    xmlfile_path = subdir_filepath
  else :
    xmlfile_path = os.path.join(metadata_topdir,samplename,f'{samplename}{PARAMETER_XMLFILE_EXT}')
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
  if EXPOSURE_XML_EXT in fp :
    xmlfile_path = fp
  else :
    if metadata_top_dir is None :
      raise RuntimeError(f'ERROR: metadata top dir must be supplied to get exposure times fo raw file path {fp}!')
    file_ext = ''
    fn_split = (os.path.basename(os.path.normpath(fp))).split(".")
    for i in range(1,len(fn_split)) :
      file_ext+=f'.{fn_split[i]}'
    sample_name = os.path.basename(os.path.dirname(os.path.normpath(fp)))
    subdir_filepath = os.path.join(metadata_top_dir,sample_name,'im3','xml',os.path.basename(os.path.normpath(fp)).replace(file_ext,EXPOSURE_XML_EXT))
    if os.path.isfile(subdir_filepath) :
      xmlfile_path = subdir_filepath
    else :
      xmlfile_path = os.path.join(metadata_top_dir,sample_name,os.path.basename(os.path.normpath(fp)).replace(file_ext,EXPOSURE_XML_EXT))
  if not os.path.isfile(xmlfile_path) :
    raise RuntimeError(f"ERROR: {xmlfile_path} searched in getExposureTimesByLayer not found!")
  root = (et.parse(xmlfile_path)).getroot()
  nlg = 0
  if nlayers==35 :
    nlg = 5
  elif nlayers==43 :
    nlg = 7
  else :
    raise ValueError(f"ERROR: number of image layers ({nlayers}) passed to getExposureTimesByLayer is not a recognized option!")
  for ilg in range(nlg) :
      layer_exposure_times_to_return+=[float(v) for v in (root[ilg].text).split()]
  return layer_exposure_times_to_return

#helper function to return a list of the median exposure times observed in each layer of a given sample
def getSampleMedianExposureTimesByLayer(metadata_topdir,samplename) :
  _,_,nlayers = getImageHWLFromXMLFile(metadata_topdir,samplename)
  if os.path.isdir(os.path.join(metadata_topdir,samplename,'im3','xml')) :
    with cd(os.path.join(metadata_topdir,samplename,'im3','xml')) :
      all_fps = [os.path.join(metadata_topdir,samplename,'im3','xml',fn) for fn in glob.glob(f'*{EXPOSURE_XML_EXT}')]
  else :
    with cd(os.path.join(metadata_topdir,samplename)) :
      all_fps = [os.path.join(metadata_topdir,samplename,fn) for fn in glob.glob(f'*{EXPOSURE_XML_EXT}')]
  utility_logger.info(f'Finding median exposure times for {samplename} ({len(all_fps)} images with {nlayers} layers each)....')
  all_exp_times_by_layer = []
  for li in range(nlayers) :
    all_exp_times_by_layer.append([])
  for fp in all_fps :
    this_image_layer_exposure_times = getExposureTimesByLayer(fp,nlayers)
    for li in range(nlayers) :
      all_exp_times_by_layer[li].append(this_image_layer_exposure_times[li])
  return np.median(np.array(all_exp_times_by_layer),1) #return the medians along the second axis

#helper function to return lists of the median exposure times and the exposure time correction offsets for all layers of a sample
def getMedianExposureTimesAndCorrectionOffsetsForSample(metadata_top_dir,samplename,et_correction_offset_file) :
  utility_logger.info("Loading info for exposure time correction...")
  median_exp_times = None; et_correction_offsets = None
  if et_correction_offset_file is not None :
    median_exp_times = getSampleMedianExposureTimesByLayer(metadata_top_dir,samplename)
    et_correction_offsets=[]
    read_layer_offsets = readtable(et_correction_offset_file,LayerOffset)
    for ln in range(1,len(median_exp_times)+1) :
      this_layer_offset = [lo.offset for lo in read_layer_offsets if lo.layer_n==ln]
      if len(this_layer_offset)==1 :
        et_correction_offsets.append(this_layer_offset[0])
      elif len(this_layer_offset)==0 :
        utility_logger.warn(f"""WARNING: LayerOffset file {et_correction_offset_file} does not have an entry for layer {ln}; offset will be set to zero!""")
        et_correction_offsets.append(0.)
      else :
        raise RuntimeError(f'ERROR: more than one entry found in LayerOffset file {et_correction_offset_file} for layer {ln}!')
  else :
    utility_logger.warn(f"""WARNING: Exposure time correction info cannot be determined from et_correction_offset_file = {et_correction_offset_file}; 
                            median exposure times and correction offsets will all be None!""")
  return median_exp_times, et_correction_offsets

#helper function to return the median exposure time and the exposure time correction offset for a given layer of a sample
def getMedianExposureTimeAndCorrectionOffsetForSampleLayer(metadata_top_dir,samplename,et_correction_offset_file,layer) :
  med_ets, et_offsets = getMedianExposureTimesAndCorrectionOffsetsForSample(metadata_top_dir,samplename,et_correction_offset_file)
  if med_ets is None or et_offsets is None :
    return None, None
  else :
    return med_ets[layer-1], et_offsets[layer-1]
