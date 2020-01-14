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

  def __init__(self, n, p1, p2, x1, y1, x2, y2, tag, layer, pscale, nclip, images):
    self.n = n
    self.p1 = p1
    self.p2 = p2
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2
    self.tag = tag

    self.layer = layer
    self.pscale = pscale
    self.nclip = nclip
    self.images = images[(p1-1,p2-1),:,:]

  def align(self):
    if self.tag % 2: return #only align edges, not corners
    self.result = AlignmentResult(
      n=self.n,
      p1=self.p1,
      p2=self.p2,
      code=self.tag,
      layer=self.layer,
    )
    self.prepimage()
    self.computeshift()
    self.shiftclip()

  def prepimage(self):
    hh, ww = self.images.shape[1:]

    #convert microns to approximate pixels
    image1x1 = floor(self.x1 * self.pscale)
    image1y1 = floor(self.y1 * self.pscale)
    image2x1 = floor(self.x2 * self.pscale)
    image2y1 = floor(self.y2 * self.pscale)
    image1x2 = image1x1 + ww
    image2x2 = image2x1 + ww
    image1y2 = image1y1 + hh
    image2y2 = image2y1 + hh

    overlapx1 = max(image1x1, image2x1)
    overlapx2 = min(image1x2, image2x2)
    overlapy1 = max(image1y1, image2y1)
    overlapy2 = min(image1y2, image2y2)

    cutimage1x1 = overlapx1 - image1x1 + self.nclip
    cutimage1x2 = overlapx2 - image1x1 - self.nclip
    cutimage1y1 = overlapy1 - image1y1 + self.nclip
    cutimage1y2 = overlapy2 - image1y1 - self.nclip

    cutimage2x1 = overlapx1 - image2x1 + self.nclip
    cutimage2x2 = overlapx2 - image2x1 - self.nclip
    cutimage2y1 = overlapy1 - image2y1 + self.nclip
    cutimage2y2 = overlapy2 - image2y1 - self.nclip

    self.cutimages = np.array([
      self.images[0,cutimage1y1:cutimage1y2,cutimage1x1:cutimage1x2],
      self.images[1,cutimage2y1:cutimage2y2,cutimage2x1:cutimage2x2],
    ])

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
  exit: int = 0
  dx: float = 0.
  dy: float = 0.
  sc: float = 0.
  mse1: float = 0.
  mse2: float = 0.
  mse3: float = 0.
  dv: float = 0.

  def __getattr__(self, attr):
    if attr == "mse":
      return self.mse1, self.mse2, self.mse3
    return super().__getattr__(attr)

  def __setattr__(self, attr, value):
    if attr == "mse":
      self.mse1, self.mse2, self.mse3 = value
      return
    super().__setattr__(attr, value)
