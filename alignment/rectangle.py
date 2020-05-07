import abc, collections, dataclasses, methodtools, numpy as np
from ..utilities import units
from ..utilities.misc import dataclass_dc_init
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
  t: int
  file: str
  pscale: dataclasses.InitVar[float]
  readingfromfile: dataclasses.InitVar[bool]

  @property
  def xvec(self):
    return np.array([self.x, self.y])

  @property
  def shape(self):
    return np.array([self.w, self.h])

@dataclass_dc_init
class ShiftedRectangle(Rectangle):
  ix: int
  iy: int
  gc: int
  px: float
  py: float
  mx1: float
  mx2: float
  my1: float
  my2: float
  gx: int
  gy: int

  def __init__(self, *args, rectangle=None, **kwargs):
    return self.__dc_init__(
      *args,
      **{
        field.name: getattr(rectangle, field.name)
        for field  in dataclasses.fields(type(rectangle))
      } if rectangle is not None else {},
      **kwargs
    )

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

@dataclasses.dataclass(frozen=True)
class ImageStats(DataClassWithDistances):
  pixelsormicrons = "microns"

  n: int
  mean: float
  min: float
  max: float
  std: float
  cx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False

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
