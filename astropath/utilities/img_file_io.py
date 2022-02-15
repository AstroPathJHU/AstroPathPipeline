#imports
import pathlib, glob, cv2, logging
import numpy as np, xml.etree.ElementTree as et
from .config import CONST
from .miscfileio import cd
from .dataclasses import MyDataClass

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

#write an array as an im3 file
def im3writeraw(outname,a) :
  with open(outname,mode='wb') as fp : #write as binary
    a.tofile(fp)

#read a raw image file and return it as an array of shape (height,width,n_layers)
def get_raw_as_hwl(fpath,height,width,nlayers,dtype=np.uint16) :
  #get the .raw file as a vector
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
  #get the file as a vector
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

class ImageLoaderBase(abc.ABC):
  #if _DEBUG is true, then when the image loader is deleted, it will print
  #a warning if its image has been loaded multiple times, for debug
  #purposes.  If __DEBUG_PRINT_TRACEBACK is also true, it will print the
  #tracebacks for each of the times the image was loaded.
  _DEBUG = True
  __DEBUG_PRINT_TRACEBACK = False

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__image_cache = None
    self.__cached_contextmanager_stack = contextlib.ExitStack()
    self.__accessed_image = False
    self.__using_image_counter = 0
    self.__debug_load_image_counter = 0
    self.__debug_load_image_traceback = []

  def __del__(self):
    if self._DEBUG:
      for i, ctr in enumerate(self.__debug_load_images_counter):
        if ctr > 1:
          for formattedtb in self.__debug_load_images_tracebacks[i]:
            printlogger("loadimage").debug("".join(formattedtb))
          warnings.warn(f"Loaded image {i} for rectangle {self} {ctr} times")

  @abc.abstractmethod
  def getimage(self):
    """
    Override this function in subclasses that actually implement
    a way of loading the image
    """

  #do not override any of these functions or call them from super()
  #override getimage() instead and call super().getimage()

  def cache_contextmanager(self, contextmanager):
    return self.__cached_contextmanager_stack.enter_context(contextmanager)

  @property
  def image(self):
    """
    Gives the HPF image.
    It gets saved in memory until you call `del rectangle.image`
    """
    self.__accessed_image = True
    return self.__image(index)
  @image.deleter
  def image(self):
    self.__accessed_image = False
    self.__check_delete_image()

  def __check_delete_images(self):
    """
    This gets called whenever you delete an image or leave a using_image context.
    It deletes images that are no longer needed in memory.
    """
    if not self.__using_image_counter and not self.__accessed_image:
      self.__image_cache = None
      self.__cached_contextmanager_stack.close()

  def __image(self):
    if self.__image_cache is None:
      self.__debug_load_image_counter += 1
      if self.__DEBUG_PRINT_TRACEBACK:
        self.__debug_load_image_tracebacks.append(traceback.format_stack())
      self.__images_cache = self.getimage()
    return self.__images_cache[i]

  @contextlib.contextmanager
  def using_image(self):
    """
    Use this in a with statement to load the image for the HPF.
    It gets freed from memory when the with statement ends.
    """
    self.__using_image_counter += 1
    try:
      yield self.__image()
    finally:
      self.__using_image_counter -= 1
      self.__check_delete_image()

class TransformedImage(ImageLoaderBase):
  def __init__(self, previmageloader, transformation):
    self.__previmageloader = previmageloader
    self.__transformation = transformation
  def getimage(self):
    with self.__previmageloader.using_image() as im:
      return self.__transformation.transform(im)

class ImageLoaderIm3Base(ImageLoaderBase):
  def __init__(self, *args, filename, usememmap=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.__filename = filename
    self.__usememmap = usememmap

  def getimage(self):
    with contextlib.ExitStack() as stack:
      enter_context = self.cache_contextmanager if self.__usememmap else stack.enter_context
      f = enter_context(open(self.__filename, "rb"))
      #use fortran order, like matlab!
      memmap = enter_context(memmapcontext(
        f,
        dtype=np.uint16,
        shape=tuple(self.imageshapeininput),
        order="F",
        mode="r"
      ))
      result = memmap.transpose(self.imagetransposefrominput)[self.imageslicefrominput]
      if not self.__usememmap:
        im = np.empty_like(result)
        im[:] = result
        result = im
      return result

  @property
  @abc.abstractmethod
  def imageshapeininput(self): pass
  @property
  @abc.abstractmethod
  def imagetransposefrominput(self): pass
  @property
  @abc.abstractmethod
  def imageslicefrominput(self): pass

class ImageLoaderIm3(ImageLoaderIm3Base):
  def __init__(self, *args, nlayers, width, height, selectlayers=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.__nlayers = nlayers
    self.__width = width
    self.__height = height
    if selectlayers is not None and not isinstance(selectlayers, (int, tuple)):
      raise TypeError(f"selectlayers should be None, an int, or a tuple, not {type(selectlayers)}")
    self.__selectlayers = selectlayers
  @property
  def imageshapeininput(self):
    return self.__nlayers, self.__width, self.__height
  @property
  def imagetransposefrominput(self):
    #it's saved as (layers, width, height), we want (height, width, layers)
    return (2, 1, 0)
  @property
  def imageslicefrominput(self):
    if self.__selectlayers is None:
      layerselection = slice(None)
    else:
      layerselection = self.__selectlayers
    return slice(None), slice(None), layerselection

class ImageLoaderIm3SingleLayer(ImageLoaderIm3Base):
  def __init__(self, *args, width, height, **kwargs):
    super().__init__(*args, **kwargs)
    self.__width = width
    self.__height = height

  @property
  def imageshapeininput(self):
    return self.__height, self.__width
  @property
  def imagetransposefrominput(self):
    #it's saved as (height, width), which is what we want
    return (0, 1)
  @property
  def imageslicefrominput(self):
    return slice(None), slice(None)

class ImageLoaderTiff(ImageLoaderBase):
  def __init__(self, *args, filename, layers, **kwargs):
    super().__init__(*args, **kwargs)
    self.__filename = filename
    self.__layers = layers

  @property
  def filename(self): return self.__filename
  @property
  def layers(self): return self.__layers

  def getimage(self):
    with tifffile.TiffFile(self.filename) as f:
      pages = f.pages
      shape, dtype = self.checktiffpages(pages)

      #make the destination array
      image = np.empty(shape=shape+(len(self.__layers),), dtype=dtype)

      #load the desired layers
      for i, layer in enumerate(self.layers):
        image[:,:,i] = pages[layer-1].asarray()

      return image

  def checktiffpages(pages):
    if not pages:
      raise ValueError(f"Tiff file {self.filename} doesn't have any pages")

    shapes = {page.shape for page in pages}
    dtypes = {page.dtype for page in pages}
    try:
      shape, = shapes
    except ValueError:
      raise ValueError(f"Tiff file {self.filename} has pages with different shapes: {shapes}")
    try:
      dtype, = dtypes
    except ValueError:
      raise ValueError(f"Tiff file {self.filename} has pages with different dtypes: {dtypes}")

    return shape, dtype

class ImageLoaderTiffSingleLayer(ImageLoaderTiff):
  def __init__(self, *args, layer, **kwargs):
    super().__init__(*args, layers=[layer], **kwargs)
  def getimage(self):
    image, = super().getimage().transpose(2, 0, 1)
    return image

class ImageLoaderComponentTiffBase(ImageLoaderTiff):
  def __init__(self, *args, nlayers, **kwargs):
    super().__init__(*args, **kwargs)
    self.__nlayers = nlayers

  @property
  def nlayers(self): return self.__nlayers

  def checktiffpages(pages):
    pagegroupslices = self.pagegroupslices
    pageindices = sorted(sum((list(range(slc.start, slc.end)) for slc in pagegroupslices), []))
    np.testing.assert_array_equal(pageindices, list(range(max(pageindices)+1)))

    npages = len(pages)
    if npages != len(pageindices):
      raise ValueError(f"Expected {len(pageindices)} pages, found {npages}")

    alllayers = range(1, npages+1)
    if set(alllayers) - set(self.layers):
      raise ValueError("Invalid layers {set(alllayers) - set(self.layers)}")

    layergroups = [list(alllayers[slc]) for slc in pagegroupslices]
    relevantgroups = (i, group for i, group in enumerate(layergroups) if set(group) & set(self.layers))
    try:
      (i, group), = relevantgroups
    except ValueError:
      raise ValueError(f"Layers are {self.layers}, expected to find layers in exactly one of these groups: {layergroups}")
    return super().checktiffpages(pages[pagegroupslices[i]])

  @property
  @abc.abstractmethod
  def pagegroupslices(self): pass

class ImageLoaderComponentTiff(ImageLoaderTiff):
  @property
  def pagegroupslices(self):
    return slice(0, self.nlayers), slice(self.nlayers, self.nlayers+1)

class ImageLoaderSegmentedComponentTiff(ImageLoaderTiff):
  def __init__(self, *args, nsegmentations, **kwargs):
    super().__init__(*args, **kwargs)
    self.__nsegmentations = nsegmentations
  @property
  def pagegroupslices(self):
    return slice(0, self.nlayers), slice(self.nlayers, self.nlayers+1), slice(self.nlayers+1, self.nlayers+1+self.nsegmentations*2)

class ImageLoaderComponentTiffSingleLayer(ImageLoaderComponentTiff, ImageLoaderTiffSingleLayer): pass
class ImageLoaderSegmentedComponentTiffSingleLayer(ImageLoaderSegmentedComponentTiff, ImageLoaderTiffSingleLayer): pass
