import abc, collections, dataclasses, numpy as np
from ..utilities import units

@dataclasses.dataclass
class Rectangle:
  n: int
  x: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: x.microns, "readfunction": float})
  y: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: x.microns, "readfunction": float})
  w: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: x.microns, "readfunction": int})
  h: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: x.microns, "readfunction": int})
  cx: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: x.microns, "readfunction": int})
  cy: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: x.microns, "readfunction": int})
  t: int
  file: str

  def setalignmentinfo(self, *, pscale):
    self.pscale = pscale
    self.x = units.Distance(microns=self.x, pscale=self.pscale)
    self.y = units.Distance(microns=self.y, pscale=self.pscale)
    self.w = units.Distance(microns=self.w, pscale=self.pscale)
    self.h = units.Distance(microns=self.h, pscale=self.pscale)
    self.cx = units.Distance(microns=self.cx, pscale=self.pscale)
    self.cy = units.Distance(microns=self.cy, pscale=self.pscale)

  @property
  def xvec(self):
    return np.array([self.x, self.y])

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
