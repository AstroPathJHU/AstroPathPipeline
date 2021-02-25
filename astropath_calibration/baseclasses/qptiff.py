import fractions, methodtools, tifffile

from ..utilities import units

class QPTiffZoomLevel(tuple, units.ThingWithQpscale):
  """
  Class that holds a zoom level of a qptiff object
  You can iterate over the 5 component image layers
    (which correspond to the broadband filters)
  """
  @methodtools.lru_cache()
  @property
  def tags(self):
    """
    The tiff tags that are common to all 5 pages.
    """
    result = {}
    for key in self[0].tags.keys():
      tags = [page.tags[key] for page in self]
      values = {tag.value for tag in tags}
      try:
        value, = values
      except ValueError:
        continue
      name, = {tag.name for tag in tags}
      result[name] = value
    return result

  @methodtools.lru_cache()
  @property
  def shape(self):
    """
    The tiff image shape
    """
    result, = {page.shape for page in self}
    return result

  @methodtools.lru_cache()
  @property
  def imagewidth(self):
    """
    The tiff image width
    """
    result, = {page.imagewidth for page in self}
    return result

  @methodtools.lru_cache()
  @property
  def resolutionunit(self):
    """
    The tiff resolution unit
    """
    return self.tags["ResolutionUnit"]
  @methodtools.lru_cache()
  @property
  def __resolutionunitdistancekeyword(self):
    """
    The keyword argument that should be given to units.Distance
    for the resolution unit
    """
    return {
      tifffile.TIFF.RESUNIT.CENTIMETER: "centimeters",
    }[self.resolutionunit]

  @methodtools.lru_cache()
  @property
  def xresolution(self):
    """
    The x resolution in pixels/micron
    """
    kw = self.__resolutionunitdistancekeyword
    xresolution = fractions.Fraction(*self.tags["XResolution"])
    return units.Distance(pixels=xresolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1)

  @methodtools.lru_cache()
  @property
  def yresolution(self):
    """
    The y resolution in pixels/micron
    """
    kw = self.__resolutionunitdistancekeyword
    yresolution = fractions.Fraction(*self.tags["YResolution"])
    return units.Distance(pixels=yresolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1)

  @methodtools.lru_cache()
  @property
  def qpscale(self):
    """
    The resolution in pixels/micron
    """
    result, = {self.xresolution, self.yresolution}
    return result

  @methodtools.lru_cache()
  @property
  def xposition(self):
    """
    The x position as a distance
    """
    kw = self.__resolutionunitdistancekeyword
    xposition = fractions.Fraction(*self.tags["XPosition"])
    return units.Distance(**{kw: xposition}, pscale=self.xresolution)

  @methodtools.lru_cache()
  @property
  def yposition(self):
    """
    The y position as a distance
    """
    kw = self.__resolutionunitdistancekeyword
    yposition = fractions.Fraction(*self.tags["YPosition"])
    return units.Distance(**{kw: yposition}, pscale=self.yresolution)

class QPTiff(tifffile.TiffFile, units.ThingWithApscale):
  """
  Class that handles a qptiff file
  """
  @property
  def zoomlevels(self):
    """
    Gives a QPTiffZoomLevel for each 5 pages in the qptiff
    """
    pages = []
    lastwidth = None
    result = []
    for page in self.pages:
      if page.tags["SamplesPerPixel"].value != 1: continue
      if page.imagewidth != lastwidth:
        lastwidth = page.imagewidth
        if pages: result.append(QPTiffZoomLevel(pages))
        pages = []
      pages.append(page)
    if pages: result.append(QPTiffZoomLevel(pages))
    return result

  @property
  def apscale(self):
    """
    Scale of the most zoomed in zoom level
    """
    return self.zoomlevels[0].qpscale

  @property
  def xposition(self):
    """
    x position of the most zoomed in zoom level
    """
    return self.zoomlevels[0].xposition
  @property
  def yposition(self):
    """
    y position of the most zoomed in zoom level
    """
    return self.zoomlevels[0].yposition
