import abc, contextlib, cv2, methodtools, numpy as np
from ..contours import findcontoursaspolygons
from .image_mask import ImageMask

class MaskLoader(abc.ABC):
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
    Read the mask for the sample and return it
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

  @methodtools.lru_cache()
  @property
  def maskpolygons(self):
    with self.using_mask() as mask:
      return findcontoursaspolygons(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, apscale=self.apscale, forgdal=True)
