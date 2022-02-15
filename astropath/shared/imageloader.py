import abc, contextlib, numpy as np, tifffile, traceback, warnings
from ..utilities.miscfileio import memmapcontext
from .logging import printlogger

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
    return self.__image()
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
    pageindices = sorted(sum((list(range(slc.start, slc.end)) for slc in pagegroupslices), []))
    np.testing.assert_array_equal(pageindices, list(range(max(pageindices)+1)))

    npages = len(pages)
    if npages != len(pageindices):
      raise ValueError(f"Expected {len(pageindices)} pages, found {npages}")

    alllayers = range(1, npages+1)
    if set(alllayers) - set(self.layers):
      raise ValueError("Invalid layers {set(alllayers) - set(self.layers)}")

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

class ImageLoaderProvideImage(ImageLoaderBase):
  def __init__(self, *args, image, **kwargs):
    self.__image = image
  def getimage(self):
    return self.__image
