import abc, contextlib, numpy as np, pathlib, PIL, tifffile, traceback, warnings
from ..utilities.miscfileio import memmapcontext
from .image_masking.image_mask import ImageMask
from .logging import printlogger

class ImageLoaderBase(abc.ABC):
  #if _DEBUG is true, then when the image loader is deleted, it will print
  #a warning if its image has been loaded multiple times, for debug
  #purposes.  If _DEBUG_PRINT_TRACEBACK is also true, it will print the
  #tracebacks for each of the times the image was loaded.

  def __init__(self, *args, _DEBUG=True, _DEBUG_PRINT_TRACEBACK=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.__image_cache = None
    self.__cached_contextmanager_stack = contextlib.ExitStack()
    self.__accessed_image = False
    self.__using_image_counter = 0
    self.__debug_load_image_counter = 0
    self.__debug_load_image_tracebacks = []
    self._DEBUG = _DEBUG
    self._DEBUG_PRINT_TRACEBACK = _DEBUG_PRINT_TRACEBACK

  def __del__(self):
    if self._DEBUG:
      if self.__debug_load_image_counter > 1:
        warnings.warn(f"Loaded image for rectangle {self} {self.__debug_load_image_counter} times")
        logger = printlogger("loadimage")
        for formattedtb in self.__debug_load_image_tracebacks:
          logger.debug("Traceback (most recent call last):")
          for item in formattedtb:
            for line in item.rstrip().split("\n"):
              logger.debug(line.replace(";", ""))
          logger.debug()

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
    return self.__image()
  @image.deleter
  def image(self):
    self.__accessed_image = False
    self.__check_delete_image()

  def __check_delete_image(self):
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
      if self._DEBUG_PRINT_TRACEBACK:
        self.__debug_load_image_tracebacks.append(traceback.format_stack())
      self.__image_cache = self.getimage()
    return self.__image_cache

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
  def __init__(self, previmageloader, transformation, **kwargs):
    super().__init__(**kwargs)
    self.__previmageloader = previmageloader
    self.__transformation = transformation
  def getimage(self):
    with self.__previmageloader.using_image() as im:
      if self.__transformation is None: return im
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
        mode="r",
      ))
      result = memmap.transpose(self.imagetransposefrominput)[self.imageslicefrominput]
      if not self.__usememmap:
        im = np.empty_like(result, order="C")
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
  @property
  def imageshapeinoutput(self):
    return np.empty(self.imageshapeininput).transpose(self.imagetransposefrominput)[self.imageslicefrominput]

class ImageLoaderIm3MultiLayer(ImageLoaderIm3Base):
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
    elif isinstance(self.__selectlayers, int):
      layerselection = self.__selectlayers - 1
    else:
      layerselection = tuple(_-1 for _ in self.__selectlayers)
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

class ImageLoaderTiffBase(ImageLoaderBase) :
  def __init__(self, *args, filename, **kwargs):
    super().__init__(*args, **kwargs)
    self.__filename = filename
  
  @property
  def filename(self): return self.__filename

  def checktiffpages(self, pages):
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

class ImageLoaderHasSingleLayerTiff(ImageLoaderTiffBase) :
  def getimage(self):
    with tifffile.TiffFile(self.filename) as f:
      pages = f.pages
      shape, dtype = self.checktiffpages(pages)
      #make the destination array
      image = np.empty(shape=shape, dtype=dtype)
      #load the single layer
      image = pages[0].asarray()
      return image

class ImageLoaderTiff(ImageLoaderTiffBase):
  def __init__(self, *args, layers, **kwargs):
    super().__init__(*args, **kwargs)
    self.__layers = layers

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

  def checktiffpages(self, pages):
    pagegroupslices = self.pagegroupslices
    pageindices = sorted(sum((list(range(slc.start, slc.stop)) for slc in pagegroupslices), []))
    np.testing.assert_array_equal(pageindices, list(range(max(pageindices)+1)))

    npages = len(pages)
    if npages != len(pageindices):
      raise ValueError(f"Expected {len(pageindices)} pages, found {npages}")

    alllayers = range(1, npages+1)
    if set(self.layers) - set(alllayers):
      raise ValueError(f"Invalid layers {set(alllayers) - set(self.layers)}")

    layergroups = [list(alllayers[slc]) for slc in pagegroupslices]
    relevantgroups = ((i, group) for i, group in enumerate(layergroups) if set(group) & set(self.layers))
    try:
      (i, group), = relevantgroups
    except ValueError:
      raise ValueError(f"Layers are {self.layers}, expected to find layers in exactly one of these groups: {layergroups}")
    return super().checktiffpages(pages[pagegroupslices[i]])

  @property
  @abc.abstractmethod
  def pagegroupslices(self): pass

class ImageLoaderComponentTiffMultiLayer(ImageLoaderComponentTiffBase):
  @property
  def pagegroupslices(self):
    return slice(0, self.nlayers), slice(self.nlayers, self.nlayers+1)

class ImageLoaderSegmentedComponentTiffMultiLayer(ImageLoaderComponentTiffBase):
  def __init__(self, *args, nsegmentations, **kwargs):
    super().__init__(*args, **kwargs)
    self.__nsegmentations = nsegmentations
  @property
  def nsegmentations(self): return self.__nsegmentations
  @property
  def pagegroupslices(self):
    return slice(0, self.nlayers), slice(self.nlayers, self.nlayers+1), slice(self.nlayers+1, self.nlayers+1+self.nsegmentations*2)

class ImageLoaderQPTiffMultiLayer(ImageLoaderTiff):
  def checktiffpages(self, pages):
    return super().checktiffpages(pages=[pages[layer-1] for layer in self.layers])

class ImageLoaderComponentTiffSingleLayer(ImageLoaderComponentTiffMultiLayer, ImageLoaderTiffSingleLayer): pass
class ImageLoaderSegmentedComponentTiffSingleLayer(ImageLoaderSegmentedComponentTiffMultiLayer, ImageLoaderTiffSingleLayer): pass
class ImageLoaderQPTiffSingleLayer(ImageLoaderQPTiffMultiLayer, ImageLoaderTiffSingleLayer): pass

class ImageLoaderNpz(ImageLoaderBase):
  def __init__(self, *args, filename, key, **kwargs):
    self.__filename = pathlib.Path(filename)
    self.__key = key
    super().__init__(*args, **kwargs)

  def getimage(self):
    dct = np.load(self.__filename)
    return dct[self.__key]

class ImageLoaderBin(ImageLoaderBase):
  def __init__(self, *args, filename, dimensions, **kwargs):
    self.__filename = pathlib.Path(filename)
    self.__dimensions = dimensions
    super().__init__(*args, **kwargs)

  def getimage(self):
    return ImageMask.unpack_tissue_mask(self.__filename, self.__dimensions)

class ImageLoaderPng(ImageLoaderBase):
  def __init__(self, *args, filename, **kwargs):
    self.__filename = pathlib.Path(filename)
    super().__init__(*args, **kwargs)

  def getimage(self):
    with PIL.Image.open(self.__filename) as im:
      return np.asarray(im)
