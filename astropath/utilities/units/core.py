"""
"""

import abc, collections, methodtools, numpy as np
from ..misc import floattoint
from ..tableio import TableReader

currentmodule = None

class UnitsError(Exception):
  """
  Exception raised when dimensional analysis gives an error
  """

class Distance:
  """
  When running in safe mode, this gives a Distance object.
  When running in fast mode, this just returns a number.

  Exactly one of these arguments is required:
    pixels: number of pixels
    microns: number of microns
    centimeters: number of centimeters
  pscale: pixels/micron scale
  power: powers of distance in the dimension (1 for a length, 2 for length^2, etc.)
  """
  def __new__(self, *args, **kwargs):
    return currentmodule.Distance(*args, **kwargs)

def onepixel(pscale):
  return Distance(pixels=1, pscale=pscale)
def onemicron(pscale):
  return Distance(microns=1, pscale=pscale)

class PScale(collections.namedtuple("PScale", ["scalename"])):
  scalename: str
  def __init__(self, scalename):
    super().__init__()
    assert self.scalename.endswith("scale"), self.scalename
  @property
  def onepixelname(self):
    if self.scalename == "pscale": return "onepixel"
    return "one"+self.scalename.replace("scale", "pixel")
  @property
  def onemicronname(self):
    if self.scalename == "pscale": return "onemicron"
    return "one"+self.scalename.replace("scale", "micron")

pscale = PScale("pscale")
qpscale = PScale("qpscale")
apscale = PScale("apscale")
imscale = PScale("imscale")

class ThingWithScale(TableReader):
  __scales = set()
  def __init_subclass__(cls, *, scale=None, scales=None, **kwargs):
    super().__init_subclass__(**kwargs)
    if scales is None: scales = []
    if scale is not None: scales.append(scale)
    cls.__scales = set.union(*(supercls.__scales for supercls in cls.__mro__ if issubclass(supercls, ThingWithScale)))
    for scale in scales:
      if isinstance(scale, str): scale = PScale(scale)

      if scale in cls.__scales: continue
      cls.__scales.add(scale)

      scalename = scale.scalename
      onepixelname = scale.onepixelname
      onemicronname = scale.onemicronname

      @abc.abstractmethod
      def scalegetter(self): return getattr(self, f"_ThingWithScale__{scalename}")
      def scalesetter(self, scale): return object.__setattr__(self, f"_ThingWithScale__{scalename}", scale)
      scalegetter.__name__ = scalesetter.__name__ = scalename
      scaleproperty = getattr(cls, scalename, property())
      if scaleproperty.fget is None: scaleproperty = scaleproperty.getter(scalegetter)
      if scaleproperty.fset is None: scaleproperty = scaleproperty.setter(scalesetter)
      setattr(cls, scalename, scaleproperty)

      def onescalepixel(self): return onepixel(pscale=getattr(self, scalename))
      onescalepixel.__name__ = onepixelname
      setattr(cls, onepixelname, methodtools.lru_cache()(property(onescalepixel)))

      def onescalemicron(self): return onemicron(pscale=getattr(self, scalename))
      onescalemicron.__name__ = onemicronname
      setattr(cls, onemicronname, methodtools.lru_cache()(property(onescalemicron)))

  def readtable(self, filename, rowclass, *, extrakwargs=None, **kwargs):
    if extrakwargs is None: extrakwargs = {}
    if issubclass(rowclass, ThingWithScale):
      for scale in self.__scales & rowclass.__scales:
        if scale.scalename in rowclass.__annotations__ and scale.scalename not in extrakwargs:
          extrakwargs[scale.scalename] = getattr(self, scale.scalename)
    return super().readtable(filename=filename, rowclass=rowclass, extrakwargs=extrakwargs, **kwargs)

class ThingWithPscale(ThingWithScale, scale="pscale"): pass
class ThingWithQpscale(ThingWithScale, scale="qpscale"): pass
class ThingWithApscale(ThingWithScale, scale="apscale"): pass
class ThingWithImscale(ThingWithPscale, ThingWithApscale, scale="imscale"):
  @property
  def ipscale(self):
    """
    The ratio of pixels/micron scales of the im3 and qptiff images
    """
    return self.pscale / self.apscale
  @property
  def ppscale(self):
    """
    The ratio of pixels/micron scales of the im3 and qptiff images,
    rounded to an integer
    """
    return floattoint(np.round(float(self.ipscale)))
  @property
  def iqscale(self):
    """
    The ratio of ipscale and ppscale, i.e. the remaining non-integer
    part of the ratio of pixels/micron scales of the im3 and qptiff
    images
    """
    return self.ipscale / self.ppscale
  @property
  def imscale(self):
    """
    The scale used for alignment of the im3 and qptiff images:
    the wsi is scaled by ppscale, which is the integer that brings
    it closest to the qptiff's scale, and the qptiff is scaled by
    whatever 1.00x is needed to bring it to the same scale.
    """
    result, = {self.apscale * self.iqscale, self.pscale / self.ppscale}
    return result
  @imscale.setter
  def imscale(self, value):
    raise AttributeError("Can't set imscale")
