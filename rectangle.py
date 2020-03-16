import dataclasses, numpy as np

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

@dataclasses.dataclass(frozen=True)
class ImageStats:
  n: int
  mean: float
  min: float
  max: float
  std: float
  cx: int
  cy: int

