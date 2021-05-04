import abc, collections, methodtools
from ..tableio import TableReader

currentmodule = None

class UnitsError(Exception): pass

class Distance:
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

class ThingWithScale(abc.ABC, TableReader):
  __scales = set()
  def __init_subclass__(cls, *, scale=None, scales=None, **kwargs):
    super().__init_subclass__(**kwargs)
    if scales is None: scales = []
    if scale is not None: scales.append(scale)
    cls.__scales = set.union(*(subcls.__scales for subcls in cls.__mro__ if issubclass(subcls, ThingWithScale)))
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
        if scale.scalename not in extrakwargs:
          extrakwargs[scale.scalename] = getattr(self, scale.scalename)
    return super().readtable(filename=filename, rowclass=rowclass, extrakwargs=extrakwargs, **kwargs)

class ThingWithPscale(ThingWithScale, scale="pscale"): pass
class ThingWithQpscale(ThingWithScale, scale="qpscale"): pass
class ThingWithApscale(ThingWithScale, scale="apscale"): pass
class ThingWithImscale(ThingWithScale, scale="imscale"): pass
