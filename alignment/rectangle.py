import abc, collections, dataclasses, numpy as np
from ..utilities.misc import dataclass_dc_init

@dataclasses.dataclass
class Rectangle:
  n: int
  x: float
  y: float
  w: int
  h: int
  cx: int
  cy: int
  t: int
  file: str

  @property
  def xvec(self):
    return np.array([self.x, self.y])

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
  @property
  def rectangledict(self):
    return rectangledict(self.rectangles)
  @property
  def rectangleindices(self):
    return {r.n for r in self.rectangles}

@dataclasses.dataclass(frozen=True)
class ImageStats:
  n: int
  mean: float
  min: float
  max: float
  std: float
  cx: int
  cy: int

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
