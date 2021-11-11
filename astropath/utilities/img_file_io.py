#imports
from .dataclasses import MyDataClass
from .miscfileio import cd
from .config import CONST
import numpy as np
import xml.etree.ElementTree as et
import pathlib, glob, cv2, logging

#global variables
PARAMETER_XMLFILE_EXT = '.Parameters.xml'
EXPOSURE_ELEMENT_NAME = 'Exposure'
CORRECTED_EXPOSURE_XML_EXT = '.Corrected.Exposure.xml'

#logger
utility_logger = logging.getLogger("utility")
utility_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s    [%(funcName)s, %(asctime)s]"))
utility_logger.addHandler(handler)

#helper class to store exposure time offset factor for a single layer (with some extra info)
class LayerOffset(MyDataClass):
  layer_n    : int
  n_overlaps : int
  offset     : float
  final_cost : float

#read the binary dump of a raw im3 file 
def im3readraw(f,dtype=np.uint16) :
  with open(f,mode='rb') as fp : #read as binary
    content = np.memmap(fp,dtype=dtype,mode='r')
  return content

#write an array of uint16s as an im3 file
def im3writeraw(outname,a) :
  with open(outname,mode='wb') as fp : #write as binary
    a.tofile(fp)

#read a raw image file and return it as an array of shape (height,width,n_layers)
def get_raw_as_hwl(fpath,height,width,nlayers,dtype=np.uint16) :
  #get the .raw file as a vector of uint16s
  try :
    img = im3readraw(fpath,dtype)
  except Exception as e :
    raise ValueError(f'ERROR: file {fpath} cannot be read as binary type {dtype}! Exception: {e}')
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

#read a single-layer image and return it as an array of shape (height,width)
def get_raw_as_hw(fpath,height,width,dtype=np.uint16) :
  #get the file as a vector of uint16s
  try :
    img = im3readraw(fpath,dtype)
  except Exception as e :
    raise ValueError(f'ERROR: file {fpath} cannot be read as binary type {dtype}! Exception: {e}')
  #reshape it
  try :
    img_a = np.reshape(img,(height,width),order="F")
  except ValueError :
    msg = f"ERROR: single layer image file ({len(img)} total bytes) shape is mismatched to"
    msg+= f" dimensions (width={width}, height={height})!"
    raise ValueError(msg)
  return img_a

#flatten and write out a given image as binary content
def write_image_to_file(img_array,filename_to_write,dtype=None) :
  #if the image is three dimensional (with the smallest dimension last, probably the number of layers), it has to be transposed
  if len(img_array.shape)==3 and (img_array.shape[2]<img_array.shape[0] and img_array.shape[2]<img_array.shape[1]) :
    img_array = img_array.transpose(2,1,0)
  elif len(img_array.shape)!=2 :
    msg = f'ERROR: writeImageToFile was passed an image of shape {img_array.shape}'
    msg+= ' instead of a 2D image, or a 3D multiplexed image with the number of layers last.'
    msg+= ' This might cause problems in writing it out in the right shape!'
    raise ValueError(msg)
  #write out image flattened in fortran order
  try :
    if dtype is not None :
      im3writeraw(filename_to_write,img_array.flatten(order="F").astype(dtype))
    else :
      im3writeraw(filename_to_write,img_array.flatten(order="F"))
  except Exception as e :
    raise RuntimeError(f'ERROR: failed to save file {filename_to_write}. Exception: {e}')

#write out each layer of a given image array as a separate image file with the same name and "*_layer_n" appended
def write_out_image_as_layers(img_array,filename_to_write,dtype=None) :
  if len(img_array.shape)!=3 :
    raise ValueError(f'ERROR: write_out_image_as_layers only takes a three-dimensional image array, but was given an array with shape {img_array.shape}!')
  for li in range(img_array.shape[-1]) :
    filename_stem = filename_to_write.split('.')[0]
    filename_ext = f'.{filename_to_write.split(".")[1]}'
    this_layer_filename = f'{filename_stem}_layer_{li+1}{filename_ext}'
    write_image_to_file(img_array[:,:,li],this_layer_filename,dtype)

#read a set of specifically-named image layer files and return them combined as a single image array
def read_image_from_layer_files(fpath,height,width,nlayers,dtype=np.uint16) :
  fname = fpath.name
  filename_stem = fname.split('.')[0]
  filename_ext = f".{fname.split('.')[1]}"
  img_array_to_return = np.empty((height,width,nlayers),dtype=dtype)
  for li in range(nlayers) :
    this_layer_filepath = fpath.parent/f'{filename_stem}_layer_{li+1}{filename_ext}'
    img_array_to_return[:,:,li] = get_raw_as_hw(this_layer_filepath,height,width,dtype)
  return img_array_to_return

#smooth an image with a Gaussian filter of the specified size
#runs on the CPU by default, but can be run on the GPU by passing gpu=True
def smooth_image_worker(im_array,smoothsigma,gpu=False) :
  if gpu :
    im_in_umat = cv2.UMat(im_array)
    im_out_umat = cv2.UMat(np.empty_like(im_array))
    cv2.GaussianBlur(im_in_umat,(0,0),smoothsigma,im_out_umat,borderType=cv2.BORDER_REPLICATE)
    return im_out_umat.get()
  else :
    return cv2.GaussianBlur(im_array,(0,0),smoothsigma,borderType=cv2.BORDER_REPLICATE)

#smooth an image and its uncertainty with a Gaussian filter of the specified size
#runs on the CPU by default, but can be run on the GPU by passing gpu=True
def smooth_image_with_uncertainty_worker(im_array,im_unc_array,smoothsigma,gpu=False) :
  ksize = 5*smoothsigma
  if ksize%2==0 :
    ksize+=1
  x_kernel = cv2.getGaussianKernel(ksize,smoothsigma)
  gaussian_kernel = (x_kernel.T)*(x_kernel)
  if gpu :
    im_in_umat = cv2.UMat(im_array); im_out_umat = cv2.UMat(np.empty_like(im_array))
    im_var_in_umat = cv2.UMat(im_unc_array**2); im_var_out_umat = cv2.UMat(np.empty_like(im_unc_array))
    cv2.filter2D(im_in_umat,cv2.CV_64F,cv2.UMat(gaussian_kernel),im_out_umat,borderType=cv2.BORDER_REPLICATE)
    cv2.filter2D(im_var_in_umat,cv2.CV_64F,cv2.UMat(gaussian_kernel**2),im_var_out_umat,borderType=cv2.BORDER_REPLICATE)
    sm_im_var = im_var_out_umat.get()
    sm_im_unc = np.where(sm_im_var>0,np.sqrt(sm_im_var),0.)
    return im_out_umat.get(),sm_im_unc
  else :
    sm_im = cv2.filter2D(im_array,cv2.CV_64F,gaussian_kernel,borderType=cv2.BORDER_REPLICATE)
    sm_im_var = cv2.filter2D(im_unc_array**2,cv2.CV_64F,gaussian_kernel**2,borderType=cv2.BORDER_REPLICATE)
    sm_im_unc = np.where(sm_im_var>0,np.sqrt(sm_im_var),0.)
    return sm_im,sm_im_unc

#get an image dimension tuple from a slide's XML file
def get_image_hwl_from_xml_file(root_dir,slideID) :
  subdir_filepath = pathlib.Path(f'{root_dir}/{slideID}/im3/xml/{slideID}{PARAMETER_XMLFILE_EXT}')
  if pathlib.Path.is_file(subdir_filepath) :
    xmlfile_path = subdir_filepath
  else :
    xmlfile_path = pathlib.Path(f'{root_dir}/{slideID}/{slideID}{PARAMETER_XMLFILE_EXT}')
  if not xmlfile_path.is_file() :
    raise FileNotFoundError(f'ERROR: xml file {xmlfile_path} does not exist!')
  try :
    tree = et.parse(xmlfile_path)
  except Exception as e :
    raise RuntimeError(f'ERROR: xml file path {xmlfile_path} could not be parsed to get image dimensions! Exception: {e}')
  for child in tree.getroot() :
    if child.attrib['name']=='Shape' :
      img_width, img_height, img_nlayers = tuple([int(val) for val in (child.text).split()])
      break
  return img_height, img_width, img_nlayers

#figure out where a raw file's exposure time xml file is given the raw file path and the root directory
def findExposureTimeXMLFile(rfp,search_dir) :
  file_ext = ''
  fn_split = ((pathlib.Path.resolve(pathlib.Path(rfp))).name).split(".")
  for i in range(1,len(fn_split)) :
    file_ext+=f'.{fn_split[i]}'
  slideID = ((pathlib.Path.resolve(pathlib.Path(rfp))).parent).name
  poss_path_1 = pathlib.Path(f'{search_dir}/{slideID}/im3/xml/{((pathlib.Path.resolve(pathlib.Path(rfp))).name).replace(file_ext,CONST.EXPOSURE_XML_EXTS[0])}')
  if pathlib.Path.is_file(poss_path_1) :
    xmlfile_path = poss_path_1
  else :
    poss_path_2 = pathlib.Path(f'{search_dir}/{slideID}/im3/xml/{((pathlib.Path.resolve(pathlib.Path(rfp))).name).replace(file_ext,CONST.EXPOSURE_XML_EXTS[1])}')
    if pathlib.Path.is_file(poss_path_2) :
      xmlfile_path = poss_path_2
    else :
      poss_path_3 = pathlib.Path(f'{search_dir}/{slideID}/{((pathlib.Path.resolve(pathlib.Path(rfp))).name).replace(file_ext,CONST.EXPOSURE_XML_EXTS[0])}')
      if pathlib.Path.is_file(poss_path_3) :
        xmlfile_path = poss_path_3
      else :
        poss_path_4 = pathlib.Path(f'{search_dir}/{slideID}/{((pathlib.Path.resolve(pathlib.Path(rfp))).name).replace(file_ext,CONST.EXPOSURE_XML_EXTS[1])}')
        if pathlib.Path.is_file(poss_path_4) :
          xmlfile_path = poss_path_4
        else :
          poss_path_5 = pathlib.Path(f'{search_dir}/{slideID}/im3/xml/{((pathlib.Path.resolve(pathlib.Path(rfp))).name).replace(file_ext,CORRECTED_EXPOSURE_XML_EXT)}')
          if pathlib.Path.is_file(poss_path_5) :
            xmlfile_path = poss_path_5    
          else :
            xmlfile_path = pathlib.Path(f'{search_dir}/{slideID}/{((pathlib.Path.resolve(pathlib.Path(rfp))).name).replace(file_ext,CORRECTED_EXPOSURE_XML_EXT)}')
  if not pathlib.Path.is_file(xmlfile_path) :
    msg = f"ERROR: findExposureTimeXMLFile could not find a valid path for raw file {rfp} given directory {search_dir}!"
    raise RuntimeError(msg)
  return xmlfile_path

#get a list of exposure times by each layer for a given raw image
#fp can be a path to a raw file or to an exposure XML file 
#but if it's a raw file the root dir must also be provided
def getExposureTimesByLayer(fp,root_dir=None) :
  layer_exposure_times_to_return = []
  if (CONST.EXPOSURE_XML_EXTS[0] in str(fp)) or (CONST.EXPOSURE_XML_EXTS[1] in str(fp)) or (CORRECTED_EXPOSURE_XML_EXT in str(fp)) :
    xmlfile_path = fp
    if not pathlib.Path.is_file(pathlib.Path(xmlfile_path)) :
      raise RuntimeError(f"ERROR: {xmlfile_path} searched in getExposureTimesByLayer not found!")
  else :
    if root_dir is None :
      raise RuntimeError(f'ERROR: root dir must be supplied to get exposure times for raw file path {fp}!')
    xmlfile_path = findExposureTimeXMLFile(fp,root_dir)
  try :
    root = (et.parse(xmlfile_path)).getroot()
  except Exception as e :
    raise RuntimeError(f'ERROR: could not parse xml file {xmlfile_path} in getExposureTimesByLayer! Exception: {e}')
  et_elements = []
  for ei in range(len(root)) :
    if 'name' in root[ei].attrib.keys() and (root[ei].attrib)['name']==EXPOSURE_ELEMENT_NAME :
      et_elements.append(root[ei])
  for et_element in et_elements :
      layer_exposure_times_to_return+=[float(v) for v in (et_element.text).split()]
  return layer_exposure_times_to_return

#return a list of the median exposure times observed in each layer of a given slide
def getSlideMedianExposureTimesByLayer(root_dir,slideID,logger=None) :
  _,_,nlayers = get_image_hwl_from_xml_file(root_dir,slideID)
  checkdir = pathlib.Path(f'{root_dir}/{slideID}/im3/xml')
  if not pathlib.Path.is_dir(checkdir) :
    checkdir = pathlib.Path(f'{root_dir}/{slideID}')
  with cd(checkdir) :
    all_fps = [pathlib.Path(f'{checkdir}/{fn}') for fn in glob.glob(f'*{CONST.EXPOSURE_XML_EXTS[0]}')]
    if len(all_fps)==0 :
      all_fps = [pathlib.Path(f'{checkdir}/{fn}') for fn in glob.glob(f'*{CONST.EXPOSURE_XML_EXTS[1]}')]
  if len(all_fps)<1 :
    raise ValueError(f'ERROR: no exposure time xml files found in directory {checkdir}!')
  msg = f'Finding median exposure times for {slideID} ({len(all_fps)} images with {nlayers} layers each)....'
  if logger is not None :
    try :
      logger.imageinfo(msg,slideID,root_dir)
    except Exception :
      try :
        logger.imageinfo(msg)
      except Exception :
        utility_logger.info(msg)
  all_exp_times_by_layer = []
  for li in range(nlayers) :
    all_exp_times_by_layer.append([])
  for fp in all_fps :
    this_image_layer_exposure_times = getExposureTimesByLayer(fp)
    for li in range(nlayers) :
      all_exp_times_by_layer[li].append(this_image_layer_exposure_times[li])
  return np.median(np.array(all_exp_times_by_layer),1) #return the medians along the second axis
