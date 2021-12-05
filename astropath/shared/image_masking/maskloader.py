import abc, contextlib, cv2, methodtools, numpy as np
from ..contours import findcontoursaspolygons
from .image_mask import ImageMask

class MaskLoader(contextlib.ExitStack, abc.ABC):
  """
  Base class for a mask that can be loaded from a file
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__using_mask_count = 0

  @abc.abstractmethod
  def maskfilename(self): pass

  def readmask(self, **filekwargs):
    """
    Read the mask and return it
    """
    filename = self.maskfilename(**filekwargs)

    filetype = filename.suffix
    if filetype == ".npz":
      dct = np.load(filename)
      return dct["mask"]
    elif filetype == ".bin":
      return ImageMask.unpack_tissue_mask(
        filename, tuple((self.ntiles * self.zoomtilesize)[::-1])
      )
    else:
      raise ValueError("Don't know how to deal with mask file type {filetype}")

  @contextlib.contextmanager
  def using_mask(self):
    """
    Context manager for using the mask.  When you enter it for the first time
    it will load the mask. If you enter it again it won't have to load it again.
    When all enters have a matching exit, it will remove it from memory.
    """
    if self.__using_mask_count == 0:
      self.__mask = self.readmask()
    self.__using_mask_count += 1
    try:
      yield self.__mask
    finally:
      self.__using_mask_count -= 1
      if self.__using_mask_count == 0:
        del self.__mask

class TissueMaskLoader(MaskLoader):
  """
  Base class for a MaskLoader that has a mask for tissue,
  which can be obtained from the main mask. (e.g. if the
  main mask has multiple classifications, the tissue mask
  could be mask == 1)
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__using_tissuemask_count = 0
    self.__using_tissuemask_uint8_count = 0

  @abc.abstractmethod
  def tissuemask(self, mask):
    """
    Get the tissue mask from the main mask
    """

  @contextlib.contextmanager
  def using_tissuemask(self):
    with contextlib.ExitStack() as stack:
      if self.__using_tissuemask_count == 0:
        self.__tissuemask = self.tissuemask(stack.enter_context(self.using_mask()))
      self.__using_tissuemask_count += 1
      try:
        yield self.__tissuemask
      finally:
        self.__using_tissuemask_count -= 1
        if self.__using_tissuemask_count == 0:
          del self.__tissuemask

  @contextlib.contextmanager
  def using_tissuemask_uint8(self):
    with contextlib.ExitStack() as stack:
      if self.__using_tissuemask_uint8_count == 0:
        self.__tissuemask_uint8 = stack.enter_context(self.using_tissuemask()).astype(np.uint8)
      self.__using_tissuemask_uint8_count += 1
      try:
        yield self.__tissuemask_uint8
      finally:
        self.__using_tissuemask_uint8_count -= 1
        if self.__using_tissuemask_uint8_count == 0:
          del self.__tissuemask_uint8

  @methodtools.lru_cache()
  @property
  def tissuemaskpolygons(self):
    with self.using_tissuemask_uint8() as mask:
      return findcontoursaspolygons(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, apscale=self.apscale, forgdal=True)
