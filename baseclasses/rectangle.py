import abc, collections, dataclasses, datetime, methodtools, numpy as np
from ..utilities import units
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

@dataclasses.dataclass
class Rectangle(DataClassWithDistances):
  pixelsormicrons = "microns"

  n: int = dataclasses.field()
  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  w: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  h: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  t: datetime.datetime = dataclasses.field(metadata={"readfunction": lambda x: datetime.datetime.fromtimestamp(int(x)), "writefunction": lambda x: int(datetime.datetime.timestamp(x))})
  file: str
  pscale: dataclasses.InitVar[float]
  readingfromfile: dataclasses.InitVar[bool]

  @property
  def xvec(self):
    return np.array([self.x, self.y])

  @property
  def cxvec(self):
    return np.array([self.cx, self.cy])

  @property
  def shape(self):
    return np.array([self.w, self.h])

@dataclasses.dataclass
class RectangleWithLayer(Rectangle):
  layer: dataclasses.InitVar[int]
  def __post_init__(self, pscale, readingfromfile, layer, *args, **kwargs):
    super().__post_init__(pscale=pscale, readingfromfile=readingfromfile, *args, **kwargs)
    self.layer = layer

class RectangleCollection(abc.ABC):
  @abc.abstractproperty
  def rectangles(self): pass
  @methodtools.lru_cache()
  def __rectangledict(self):
    return rectangledict(self.rectangles)
  @property
  def rectangledict(self): return self.__rectangledict()
  @property
  def rectangleindices(self):
    return {r.n for r in self.rectangles}

class RectangleList(list, RectangleCollection):
  @property
  def rectangles(self): return self

def rectangledict(rectangles):
  return {rectangle.n: i for i, rectangle in enumerate(rectangles)}

def rectangleoroverlapfilter(selection, *, compatibility=False):
  if compatibility:
    if selection == -1:
      selection = None
    if isinstance(selection, tuple):
      if len(selection) == 2:
        selection = range(selection[0], selection[1]+1)
      else:
        selection = str(selection) #to get the right error message below

  if selection is None:
    return lambda r: True
  elif isinstance(selection, collections.abc.Collection) and not isinstance(selection, str):
    return lambda r: r.n in selection
  elif isinstance(selection, collections.abc.Callable):
    return selection
  else:
    raise ValueError(f"Unknown rectangle or overlap selection: {selection}")

rectanglefilter = rectangleoroverlapfilter
