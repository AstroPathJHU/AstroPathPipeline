import dataclasses

@dataclasses.dataclass
class Overlap:
  n: int
  p1: int
  p2: int
  x1: float
  y1: float
  x2: float
  y2: float
  tag: int

  def align(self):
    if self.tag % 2: return #only align edges, not corners
    self.prepimage()
    self.computeshift()
    self.shiftclip()

  def prepimage(self):
    raise NotImplementedError
  def computeshift(self):
    raise NotImplementedError
  def shiftclip(self):
    raise NotImplementedError

@dataclasses.dataclass
class AlignmentResult:
  n: int
  p1: int
  p2: int
  code: int
  layer: int
  exit: int
  dx: float
  dy: float
  sc: float
  mse1: float
  mse2: float
  mse3: float
  dv: float


