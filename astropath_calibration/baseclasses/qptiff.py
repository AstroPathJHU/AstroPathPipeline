import fractions, methodtools, tifffile

from ..utilities import units

class QPTiffZoomLevel(tuple):
  @methodtools.lru_cache()
  @property
  def tags(self):
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
    result, = {page.shape for page in self}
    return result

  @methodtools.lru_cache()
  @property
  def imagewidth(self):
    result, = {page.imagewidth for page in self}
    return result

  @methodtools.lru_cache()
  @property
  def resolutionunit(self):
    return self.tags["ResolutionUnit"]
  @methodtools.lru_cache()
  @property
  def resolutionunitdistancekeyword(self):
    return {
      tifffile.TIFF.RESUNIT.CENTIMETER: "centimeters",
    }[self.resolutionunit]

  @methodtools.lru_cache()
  @property
  def xresolution(self):
    kw = self.resolutionunitdistancekeyword
    xresolution = fractions.Fraction(*self.tags["XResolution"])
    return units.Distance(pixels=xresolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1)

  @methodtools.lru_cache()
  @property
  def yresolution(self):
    kw = self.resolutionunitdistancekeyword
    yresolution = fractions.Fraction(*self.tags["YResolution"])
    return units.Distance(pixels=yresolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1)

  @methodtools.lru_cache()
  @property
  def qpscale(self):
    result, = {self.xresolution, self.yresolution}
    return result

  @methodtools.lru_cache()
  @property
  def xposition(self):
    kw = self.resolutionunitdistancekeyword
    xposition = fractions.Fraction(*self.tags["XPosition"])
    return units.Distance(**{kw: xposition}, pscale=self.xresolution)

  @methodtools.lru_cache()
  @property
  def yposition(self):
    kw = self.resolutionunitdistancekeyword
    yposition = fractions.Fraction(*self.tags["YPosition"])
    return units.Distance(**{kw: yposition}, pscale=self.yresolution)

class QPTiff(tifffile.TiffFile):
  @property
  def zoomlevels(self):
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
    return self.zoomlevels[0].qpscale

  @property
  def xposition(self):
    return self.zoomlevels[0].xposition
  @property
  def yposition(self):
    return self.zoomlevels[0].yposition
