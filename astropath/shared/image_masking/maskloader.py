import abc, collections, contextlib, cv2, methodtools, numpy as np, scipy.ndimage, skimage.measure, skimage.transform
from ...utilities import units
from ..contours import findcontoursaspolygons
from ..logging import ThingWithLogger
from .image_mask import ImageMask

class MaskLoader(contextlib.ExitStack):
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

class TissueMaskLoaderWithPolygons(TissueMaskLoader, ThingWithLogger, contextlib.ExitStack):
  @methodtools.lru_cache()
  def __tissuemaskpolygons_and_area_cutoff(self, *, zoomfactor, epsilon):
    self.logger.debug("finding tissue mask as polygons")
    self.logger.debug("  loading mask")
    with self.using_tissuemask_zoomed(zoomfactor) as mask:
      imagescale = self.pscale/zoomfactor
      self.logger.debug("  cropping the mask to regions that contain tissue")
      xindices, yindices = np.where(mask)
      minx = np.min(xindices)
      maxx = np.max(xindices)
      miny = np.min(yindices)
      maxy = np.max(yindices)
      croppedmask = mask[minx:maxx+1, miny:maxy+1]
      self.logger.debug("  creating negative mask")
      notmask = ~croppedmask
      self.logger.debug("  filling holes to find outer regions")
      filled = scipy.ndimage.binary_fill_holes(croppedmask)
      self.logger.debug("  finding outer regions")
      labeled, nlabels = scipy.ndimage.label(filled, structure=[[0,1,0],[1,1,1],[0,1,0]])
      self.logger.debug("finding region properties")
      properties = skimage.measure.regionprops(labeled)
      self.logger.debug("  finding areas of outer regions")
      totalarea = sum(props.area for props in properties)
      areacutoff = 0.001 * totalarea
      properties.sort(key=lambda x: x.area, reverse=True)
      for idx, props in enumerate(properties):
        self.logger.debug(f"  looking at outer region {idx+1} / {nlabels}")
        if idx >= 100 or props.area < areacutoff: #drop the tiny labels, and only keep 100 labels maximum
          self.logger.debug(f"  too small (rank {idx+1}, area {props.area}) --> skip it")
          croppedmask[props.slice][props.image] = False
          continue

        #now we have a large region
        #fill in the tiny holes
        self.logger.debug("  filling in the tiny holes")
        holes = props.image & notmask[props.slice]
        labeledholes, nholelabels = scipy.ndimage.label(holes)
        holeproperties = skimage.measure.regionprops(labeledholes)
        for holeprops in holeproperties:
          if holeprops.area / props.area < 0.0025:
            croppedmask[props.slice][holeprops.slice][holeprops.image] = True

      self.logger.debug("converting to gdal")
      mask = mask.astype(np.uint8)
      polygons = findcontoursaspolygons(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, annoscale=self.pscale, imagescale=imagescale, forgdal=True)
      areacutoff = units.convertpscale(areacutoff * units.onepixel(imagescale)**2, imagescale, self.pscale, power=2)

      self.logger.debug("smoothing")
      polygons = [p.smooth_rdp(epsilon=epsilon) for p in polygons]

      return polygons, areacutoff

  def tissuemaskpolygons(self, *, zoomfactor=1, epsilon=None):
    """
    Get the outline of the tissue mask as gdal polygons.

    zoomfactor: zoom in by this amount for calculating the polygons
                (however the results can be off by zoomfactor*1 pixel)
                default: 1 (no zoom)
    epsilon: Smooth the polygon with the Ramer-Douglas-Peucker algorithm
             with this epsilon (default: 2 pixels)
    """
    if epsilon is None: epsilon = 2*self.onepixel
    return self.__tissuemaskpolygons_and_area_cutoff(zoomfactor=zoomfactor, epsilon=epsilon)[0]
  def tissuemaskpolygonareacutoff(self, *, zoomfactor=1, epsilon=None):
    if epsilon is None: epsilon = 2*self.onepixel
    return self.__tissuemaskpolygons_and_area_cutoff(zoomfactor=zoomfactor, epsilon=epsilon)[1]
