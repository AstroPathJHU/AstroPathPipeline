import collections, fractions, methodtools, numpy as np, pathlib, tifffile

from ..utilities import units
from .imageloader import ImageLoaderQPTiffMultiLayer, ImageLoaderQPTiffSingleLayer

class QPTiffZoomLevel(collections.abc.Sequence, units.ThingWithQpscale):
  """
  Class that holds a zoom level of a qptiff object
  You can iterate over the 5 component image layers
    (which correspond to the broadband filters)
  """
  def __init__(self, *, filename, pages, pageindices):
    self.__filename = pathlib.Path(filename)
    self.__pages = pages
    self.__pageindices = pageindices
  def __getitem__(self, item):
    return self.__pages[item]
  def __len__(self):
    return len(self.__pages)
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
    return units.Distance(pixels=float(xresolution), pscale=1) / units.Distance(**{kw: 1}, pscale=1)

  @methodtools.lru_cache()
  @property
  def yresolution(self):
    """
    The y resolution in pixels/micron
    """
    kw = self.__resolutionunitdistancekeyword
    yresolution = fractions.Fraction(*self.tags["YResolution"])
    return units.Distance(pixels=float(yresolution), pscale=1) / units.Distance(**{kw: 1}, pscale=1)

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

  @property
  def position(self):
    return np.array([self.xposition, self.yposition])

  @methodtools.lru_cache()
  def __imageloader(self, *, layers, layer):
    if (layers is None) + (layer is None) != 1:
      raise ValueError("Have to provide exactly one of layers or layer")

    for layer in (layers if layers is not None else [layer]):
      if not (1 <= layer <= len(self)):
        raise ValueError(f"Invalid layer {layer}, has to be between 1 and {len(self)}")

    kwargs = {"filename": self.__filename}
    if layers is not None:
      return ImageLoaderQPTiffMultiLayer(**kwargs, layers=[self.__pageindices[layer-1] for layer in layers])
    else:
      return ImageLoaderQPTiffSingleLayer(**kwargs, layer=self.__pageindices[layer-1])

  def imageloader(self, *, layers=None, layer=None):
    """
    Multiple layers of functions to make the lru_cache work
    """
    return self.__imageloader(layers=layers, layer=layer)

  def using_image(self, *, layers=None, layer=None):
    return self.imageloader(layers=layers, layer=layer).using_image()

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
    pageindices = []
    lastwidth = None
    result = []
    for i, page in enumerate(self.pages, start=1):
      if page.tags["SamplesPerPixel"].value != 1: continue
      if page.imagewidth != lastwidth:
        lastwidth = page.imagewidth
        if pages: result.append(QPTiffZoomLevel(filename=self.filehandle.path, pages=pages, pageindices=pageindices))
        pages = []
        pageindices = []
      pages.append(page)
      pageindices.append(i)
    if pages: result.append(QPTiffZoomLevel(filename=self.filehandle.path, pages=pages, pageindices=pageindices))
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

  @property
  def position(self):
    """
    position of the most zoomed in zoom level
    """
    return np.array([self.xposition, self.yposition])
