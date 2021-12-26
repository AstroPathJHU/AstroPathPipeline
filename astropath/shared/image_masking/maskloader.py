import abc, collections, contextlib, cv2, methodtools, numpy as np, scipy.ndimage, skimage.transform
from ...utilities import units
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

class TissueMaskLoader(units.ThingWithPscale, MaskLoader):
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
    self.__using_tissuemask_zoomed_count = collections.defaultdict(lambda: 0)

    self.__tissuemask_zoomed = {}

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

  @contextlib.contextmanager
  def using_tissuemask_zoomed(self, zoomfactor):
    if zoomfactor == 1:
      with self.using_tissuemask() as mask:
        yield mask
      return

    with contextlib.ExitStack() as stack:
      if self.__using_tissuemask_zoomed_count[zoomfactor] == 0:
        self.__tissuemask_zoomed[zoomfactor] = (skimage.transform.downscale_local_mean(stack.enter_context(self.using_tissuemask()), (zoomfactor, zoomfactor)) > 0.5)
      self.__using_tissuemask_zoomed_count[zoomfactor] += 1
      try:
        yield self.__tissuemask_zoomed[zoomfactor]
      finally:
        self.__using_tissuemask_zoomed_count[zoomfactor] -= 1
        if self.__using_tissuemask_zoomed_count[zoomfactor] == 0:
          del self.__tissuemask_zoomed[zoomfactor]

class TissueMaskLoaderWithPolygons(TissueMaskLoader, units.ThingWithAnnoscale):
  @methodtools.lru_cache()
  def __tissuemaskpolygons_and_area_cutoff(self, *, zoomfactor):
    with self.using_tissuemask_zoomed(zoomfactor) as mask:
      imagescale = self.pscale/zoomfactor
      notmask = ~mask
      filled = scipy.ndimage.binary_fill_holes(mask)
      labeled, nlabels = scipy.ndimage.label(filled, structure=[[0,1,0],[1,1,1],[0,1,0]])
      labels = range(1, nlabels+1)
      areas = {label: np.count_nonzero(labeled==label) for label in labels}
      totalarea = sum(areas.values())
      areacutoff = 0.001 * totalarea
      labels = sorted(labels, key=areas.get, reverse=True)
      for idx, label in reversed(list(enumerate(labels))):
        area = areas[label]
        if idx >= 100: #only keep 100 labels maximum
          mask[labeled==label] = 0
          continue
        elif area < areacutoff: #drop the tiny ones
          mask[labeled==label] = 0
          continue

        #now we have a large region
        #fill in the tiny holes
        holes = (labeled == label) & notmask
        labeledholes, nholelabels = scipy.ndimage.label(holes)
        holelabels = range(1, nholelabels+1)
        for holelabel in holelabels:
          hole = labeledholes == holelabel
          holearea = np.count_nonzero(hole)
          if holearea / area < 0.0025:
            mask[hole] = True

      mask = mask.astype(np.uint8)
      polygons = findcontoursaspolygons(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, annoscale=self.annoscale, imagescale=imagescale, forgdal=True)
      areacutoff = units.convertpscale(areacutoff * units.onepixel(imagescale)**2, imagescale, self.pscale, power=2)
      return polygons, areacutoff

  def tissuemaskpolygons(self, *, zoomfactor=16):
    return self.__tissuemaskpolygons_and_area_cutoff(zoomfactor=zoomfactor)[0]
  def tissuemaskpolygonareacutoff(self, *, zoomfactor=16):
    return self.__tissuemaskpolygons_and_area_cutoff(zoomfactor=zoomfactor)[1]
