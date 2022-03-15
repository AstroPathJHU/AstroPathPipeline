import abc, contextlib, cv2, methodtools, numpy as np, scipy.ndimage, skimage.measure, skimage.transform
from ...utilities import units
from ..contours import findcontoursaspolygons
from ..imageloader import TransformedImage
from ..logging import ThingWithLogger

class ThingWithMask(abc.ABC):
  @property
  @abc.abstractmethod
  def maskloader(self): pass
  def using_mask(self): return self.maskloader.using_image()

class ThingWithTissueMask(ThingWithMask):
  @property
  @abc.abstractmethod
  def tissuemasktransformation(self): pass
  @methodtools.lru_cache()
  @property
  def tissuemaskloader(self):
    return TransformedImage(self.maskloader, self.tissuemasktransformation, _DEBUG_PRINT_TRACEBACK=True)
  def using_tissuemask(self): return self.tissuemaskloader.using_image()

class ThingWithTissueMaskPolygons(ThingWithTissueMask, ThingWithLogger, contextlib.ExitStack):
  @methodtools.lru_cache()
  def __tissuemaskpolygons_and_area_cutoff(self, *, epsilon):
    self.logger.debug("finding tissue mask as polygons")
    self.logger.debug("  loading mask")
    with self.using_tissuemask() as mask:
      imagescale = self.pscale
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

  def tissuemaskpolygons(self, *, epsilon=None):
    """
    Get the outline of the tissue mask as gdal polygons.

    epsilon: Smooth the polygon with the Ramer-Douglas-Peucker algorithm
             with this epsilon (default: 2 pixels)
    """
    if epsilon is None: epsilon = 2*self.onepixel
    return self.__tissuemaskpolygons_and_area_cutoff(epsilon=epsilon)[0]
  def tissuemaskpolygonareacutoff(self, *, epsilon=None):
    if epsilon is None: epsilon = 2*self.onepixel
    return self.__tissuemaskpolygons_and_area_cutoff(epsilon=epsilon)[1]
