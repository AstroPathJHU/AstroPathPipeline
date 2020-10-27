#imports
from .tableio import readtable
from .misc import cd
import numpy as np
import xml.etree.ElementTree as et
import os, glob, cv2, logging, dataclasses, time

#global variables
PARAMETER_XMLFILE_EXT = '.Parameters.xml'
EXPOSURE_XML_EXT      = '.SpectralBasisInfo.Exposure.xml'
CORRECTED_EXPOSURE_XML_EXT = '.Corrected.Exposure.xml'
XML_START_LINE_TEXT = '<IM3Fragment>'
XML_END_LINE_TEXT = '</IM3Fragment>'
GENERATED_BY_LINE_TEXT = 'This XML file was generated by'
GENERATED_BY_TO_WRITE = 'correct_and_copy_rawfiles.py'
DATE_LINE_TEXT = 'Date : '
FILE_LINE_TEXT = 'File : '
ORIGINAL_XML_FILE_LINE_TEXT = 'Original .xml File : '
HEADER_LINE_END = '--->'

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
  try :
    img = im3readraw(fname,dtype)
  except Exception as e :
    raise ValueError(f'ERROR: file {fname} cannot be read as binary type {dtype}! Exception: {e}')
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
  try :
    img = im3readraw(fname,dtype)
  except Exception as e :
    raise ValueError(f'ERROR: file {fname} cannot be read as binary type {dtype}! Exception: {e}')
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
  try :
    im3writeraw(filename_to_write,img_array.flatten(order="F").astype(dtype))
  except Exception as e :
    raise RuntimeError(f'ERROR: failed to save file {filename_to_write}. Exception: {e}')

#helper function to smooth an image
#this can be run in parallel on the GPU
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
  try :
    tree = et.parse(xmlfile_path)
  except Exception as e :
    raise RuntimeError(f'ERROR: xml file path {xmlfile_path} could not be parsed to get image dimensions! Exception: {e}')
  for child in tree.getroot() :
    if child.attrib['name']=='Shape' :
      img_width, img_height, img_nlayers = tuple([int(val) for val in (child.text).split()])
  return img_height, img_width, img_nlayers

#helper function to figure out where a raw file's exposure time xml file is given the raw file path and the metadata directory
def findExposureTimeXMLFile(rfp,search_dir) :
  file_ext = ''
  fn_split = (os.path.basename(os.path.normpath(rfp))).split(".")
  for i in range(1,len(fn_split)) :
    file_ext+=f'.{fn_split[i]}'
  sample_name = os.path.basename(os.path.dirname(os.path.normpath(rfp)))
  subdir_filepath_1 = os.path.join(search_dir,sample_name,'im3','xml',os.path.basename(os.path.normpath(rfp)).replace(file_ext,EXPOSURE_XML_EXT))
  if os.path.isfile(subdir_filepath_1) :
    xmlfile_path = subdir_filepath_1
  else :
    subdir_filepath_2 = os.path.join(search_dir,sample_name,'im3','xml',os.path.basename(os.path.normpath(rfp)).replace(file_ext,CORRECTED_EXPOSURE_XML_EXT))
    if os.path.isfile(subdir_filepath_2) :
      xmlfile_path = subdir_filepath_2
    else :
      other_path = os.path.join(search_dir,sample_name,os.path.basename(os.path.normpath(rfp)).replace(file_ext,EXPOSURE_XML_EXT))
      if os.path.isfile(other_path) :
        xmlfile_path = other_path
      else :
        xmlfile_path = os.path.join(search_dir,sample_name,os.path.basename(os.path.normpath(rfp)).replace(file_ext,CORRECTED_EXPOSURE_XML_EXT))
  if not os.path.isfile(xmlfile_path) :
    msg = f"ERROR: findExposureTimeXMLFile could not find a valid path for raw file {rfp} given directory {search_dir}!"
    msg+= f' (None of {subdir_filepath_1}, {subdir_filepath_2}, {other_path}, and {xmlfile_path} exist!)'
    raise RuntimeError(msg)
  return xmlfile_path

#helper function to write out a new exposure time xml file with the layer exposure times replaced
def writeModifiedExposureTimeXMLFile(infile_path,new_ets,edit_header=False) :
  if not os.path.isfile(infile_path) :
    raise FileNotFoundError(f'ERROR: original exposure time .xml file path {infile_path} does not exist!')
  #get all the lines from the original file
  try :
    with open(infile_path,'r') as ifp :
      all_lines = [l.rstrip() for l in ifp.readlines()]
  except Exception as e :
    raise ValueError(f'ERROR: could not read original exposure time .xml file path {infile_path}. Exception: {e}')
  line_index = 0    
  #make the new header info
  new_header_lines = []
  while (not all_lines[line_index].startswith(XML_START_LINE_TEXT)) :
    line = all_lines[line_index]
    if edit_header :
      if GENERATED_BY_LINE_TEXT in line :
        to_replace = ((((line.split(GENERATED_BY_LINE_TEXT))[-1]).lstrip()).rstrip(HEADER_LINE_END)).rstrip()
        new_header_lines.append(line.replace(to_replace,GENERATED_BY_TO_WRITE))
      elif DATE_LINE_TEXT in line :
        to_replace = ((((line.split(DATE_LINE_TEXT))[-1]).lstrip()).rstrip(HEADER_LINE_END)).rstrip()
        new_header_lines.append(line.replace(to_replace,time.strftime("%Y-%m-%d %H:%M:%S")))
      elif FILE_LINE_TEXT in line :
        new_header_lines.append(line)
        to_replace = ((((line.split(FILE_LINE_TEXT))[-1]).lstrip()).rstrip(HEADER_LINE_END)).rstrip()
        orig_xml_file_line = line.replace(FILE_LINE_TEXT,ORIGINAL_XML_FILE_LINE_TEXT)
        orig_xml_file_line = orig_xml_file_line.replace(to_replace,os.path.abspath(infile_path))
        new_header_lines.append(orig_xml_file_line)
      else :
        new_header_lines.append(line)
    else :
        new_header_lines.append(line)
    line_index+=1
  #make the new xml info
  new_xml_lines = []
  old_elem_lines = []; old_data_texts = []; new_data_texts = []
  attrib_names = ['name','type','size']
  try :
    tree = et.parse(infile_path)
  except Exception as e :
    raise ValueError(f'ERROR: could not parse original exposure time .xml file {infile_path}. Exception: {e}')
  iet = 0
  for elem in tree.getroot() :
    old_elem_line = '<D '
    attributes = elem.attrib
    keys = attributes.keys()
    for an in attrib_names :
      if an in keys :
        old_elem_line+=f'{an}="{attributes[an]}" '
      else :
        msg = f'Expected key "{an}" not found in XML element attributes {attributes} for {infile_path}!'
        raise RuntimeError(msg)
    old_elem_line=old_elem_line[:-1]+'>'
    old_data = elem.text
    old_elem_line=old_elem_line+f'{old_data}</D>'
    old_elem_lines.append(old_elem_line)
    old_data_texts.append(old_data)
    new_data = ''
    for v in old_data.split() :
      n_dec_places = len(v.split('.')[-1])
      new_data+=f'{new_ets[iet]:.{n_dec_places}f} '
      iet+=1
    new_data = new_data[:-1]
    new_data_texts.append(new_data)
  if iet!=len(new_ets) :
    msg = f'ERROR: {len(new_ets)} new exposure times sent to writeModifiedExposureTimeXMLFile but '
    msg+= f'input file {infile_path} only contains {iet}' 
    raise RuntimeError(msg)
  while line_index<len(all_lines) :
    line = all_lines[line_index]
    if line.startswith(XML_START_LINE_TEXT) or line.startswith(XML_END_LINE_TEXT) :
      new_xml_lines.append(line)
    else :
      try :
        ei = old_elem_lines.index(line.lstrip())
      except ValueError :
        msg = f'ERROR: text file line "{line.lstrip}" does not correspond to any of the XML elements '
        msg+= f'found in {infile_path}!'
        raise RuntimeError(msg)
      new_xml_lines.append(line.replace(old_data_texts[ei],new_data_texts[ei]))
    line_index+=1
  #write out the new file
  all_new_lines = new_header_lines+new_xml_lines
  try :
    outfile_name = os.path.basename(os.path.normpath(infile_path)).replace(EXPOSURE_XML_EXT,CORRECTED_EXPOSURE_XML_EXT)
    with open(outfile_name,'w') as ofp :
      for il,line in enumerate(all_new_lines) :
        if il<len(all_new_lines)-1 :
          ofp.write(f'{line}\n')
        else :
          ofp.write(f'{line}')
  except Exception as e :
    raise RuntimeError(f'ERROR: could not write out modified exposure time .xml file. Exception: {e}')
  utility_logger.info(f'Wrote out new exposure time xml file {outfile_name}')

#helper function to get a list of exposure times by each layer for a given raw image
#fp can be a path to a raw file or to an exposure XML file 
#but if it's a raw file the metadata top dir must also be provided
def getExposureTimesByLayer(fp,nlayers,metadata_top_dir=None) :
  layer_exposure_times_to_return = []
  if (EXPOSURE_XML_EXT in fp) or (CORRECTED_EXPOSURE_XML_EXT in fp) :
    xmlfile_path = fp
    if not os.path.isfile(xmlfile_path) :
      raise RuntimeError(f"ERROR: {xmlfile_path} searched in getExposureTimesByLayer not found!")
  else :
    if metadata_top_dir is None :
      raise RuntimeError(f'ERROR: metadata top dir must be supplied to get exposure times for raw file path {fp}!')
    xmlfile_path = findExposureTimeXMLFile(fp,metadata_top_dir)
  try :
    root = (et.parse(xmlfile_path)).getroot()
  except Exception as e :
    raise RuntimeError(f'ERROR: could not parse xml file {xmlfile_path} in getExposureTimesByLayer! Exception: {e}')
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
  checkdir = os.path.join(metadata_topdir,samplename,'im3','xml')
  if not os.path.isdir(checkdir) :
    checkdir = os.path.join(metadata_topdir,samplename)
  with cd(checkdir) :
    all_fps = [os.path.join(checkdir,fn) for fn in glob.glob(f'*{EXPOSURE_XML_EXT}')]
  if len(all_fps)<1 :
    raise ValueError(f'ERROR: no exposure time xml files found in directory {checkdir}!')
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
  if not os.path.isfile(et_correction_offset_file) :
    raise FileNotFoundError(f'ERROR: Exposure time correction info cannot be determined from et_correction_offset_file = {et_correction_offset_file}!')
  utility_logger.info("Loading info for exposure time correction...")
  median_exp_times = getSampleMedianExposureTimesByLayer(metadata_top_dir,samplename)
  et_correction_offsets=[]
  try :
    read_layer_offsets = readtable(et_correction_offset_file,LayerOffset)
  except Exception as e :
    msg = f'ERROR: could not read {et_correction_offset_file} as a LayerOffset file in getMedianExposureTimesAndCorrectionOffsetsForSample. Exception: {e}'
    raise RuntimeError(msg)
  for ln in range(1,len(median_exp_times)+1) :
    this_layer_offset = [lo.offset for lo in read_layer_offsets if lo.layer_n==ln]
    if len(this_layer_offset)==1 :
      et_correction_offsets.append(this_layer_offset[0])
    elif len(this_layer_offset)==0 :
      utility_logger.warn(f'WARNING: LayerOffset file {et_correction_offset_file} does not have an entry for layer {ln}; offset will be set to zero!')
      et_correction_offsets.append(0.)
    else :
      raise RuntimeError(f'ERROR: more than one entry found in LayerOffset file {et_correction_offset_file} for layer {ln}!')
  return median_exp_times, et_correction_offsets

#helper function to return the median exposure time and the exposure time correction offset for a given layer of a sample
def getMedianExposureTimeAndCorrectionOffsetForSampleLayer(metadata_top_dir,samplename,et_correction_offset_file,layer) :
  med_ets, et_offsets = getMedianExposureTimesAndCorrectionOffsetsForSample(metadata_top_dir,samplename,et_correction_offset_file) 
  med_et = med_ets[layer-1]; et_offset = et_offsets[layer-1]
  return med_et, et_offset
