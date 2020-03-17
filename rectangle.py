import abc, dataclasses, numpy as np

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

class RectangleCollection(abc.ABC):
  @abc.abstractproperty
  def rectangles(self): pass
  @property
  def rectangledict(self):
    return rectangledict(self.rectangles)

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

