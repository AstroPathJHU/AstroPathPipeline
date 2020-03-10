import abc, dataclasses, matplotlib.pyplot as plt, numpy as np, uncertainties as unc

from .computeshift import computeshift, mse, shiftimg

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

  def setalignmentinfo(self, layer, pscale, nclip, rectangles):
    self.layer = layer
    self.pscale = pscale
    self.nclip = nclip
    self.rectangles = rectangles
    self.result = None

  @property
  def images(self):
    result = tuple(r.image[:] for r in self.rectangles)
    for i in result: i.flags.writeable = False
    return result

  @property
  def cutimages(self):
    image1, image2 = self.images

    hh, ww = image1.shape
    assert (hh, ww) == image2.shape

    #convert microns to approximate pixels
    image1x1 = int(self.x1 * self.pscale)
    image1y1 = int(self.y1 * self.pscale)
    image2x1 = int(self.x2 * self.pscale)
    image2y1 = int(self.y2 * self.pscale)
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

    return (
      image1[cutimage1y1:cutimage1y2,cutimage1x1:cutimage1x2],
      image2[cutimage2y1:cutimage2y2,cutimage2x1:cutimage2x2],
    )

  def align(self, *, debug=False, alreadyalignedstrategy="error", **computeshiftkwargs):
    if self.result is not None:
      if alreadyalignedstrategy == "error":
        raise RuntimeError(f"Overlap {self.n} is already aligned.  To keep the previous result, call align(alreadyalignedstrategy='skip').  To align again and overwrite the previous result, call align(alreadyalignedstrategy='overwrite').")
      elif alreadyalignedstrategy == "skip":
        return self.result
      elif alreadyalignedstrategy == "overwrite":
        pass
      else:
        raise ValueError(f"Unknown value alreadyalignedstrategy={alreadyalignedstrategy!r}")
    self.result = AlignmentResult(
      n=self.n,
      p1=self.p1,
      p2=self.p2,
      code=self.tag,
      layer=self.layer,
      exit=-1,
    )
    try:
      self.__computeshift(**computeshiftkwargs)
      self.__shiftclip()
    except Exception as e:
      self.result.exit = 3
      self.result.dxdy = unc.ufloat(0, 9999), unc.ufloat(0, 9999)
      self.result.sc = 1.
      self.result.exception = e
      if debug: raise
    else:
      self.result.exception = None
    return self.result

  def getinversealignment(self, inverse):
    assert (inverse.p1, inverse.p2) == (self.p2, self.p1)
    self.result = AlignmentResult(
      n = self.n,
      p1 = self.p1,
      p2 = self.p2,
      code = self.tag,
      layer = self.layer,
      exit = inverse.result.exit,
      dx = -inverse.result.dx,
      dy = -inverse.result.dy,
      sc = 1/inverse.result.sc,
      mse1 = inverse.result.mse2,
      mse2 = inverse.result.mse1,
      mse3 = inverse.result.mse3 / inverse.result.sc**2,
    )
    self.result.covariance = inverse.result.covariance
    return self.result

  def __computeshift(self, **computeshiftkwargs):
    minimizeresult = computeshift(self.cutimages, **computeshiftkwargs)
    self.result.dxdy = minimizeresult.dx, minimizeresult.dy
    self.result.exit = minimizeresult.exit

  @property
  def shifted(self):
    return shiftimg(self.cutimages, self.result.dx, self.result.dy)

  def __shiftclip(self):
    """
    Shift images symetrically by fractional amount
    and save the result. Compute the mse and the
    illumination correction
    """
    b1, b2 = self.shifted

    mse1 = mse(b1)
    mse2 = mse(b2)

    self.result.sc = (mse1 / mse2) ** 0.5

    diff = b1 - b2*self.result.sc
    self.result.mse = mse1, mse2, mse(diff)

  def getimage(self,normalize=100.,shifted=True) :
    if shifted:
      red, green = self.shifted
    else:
      red, green = self.cutimages
    blue = (red+green)/2
    img = np.array([red, green, blue]).transpose(1, 2, 0) / normalize
    return img

  def showimages(self, normalize=100., shifted=True):
    img=self.getimage(normalize,shifted)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

  def getShiftComparisonImageCodeNameTuple(self) :
    return (self.result.code,f'overlap_{self.result.n}_[{self.result.p1}x{self.result.p2},type{self.result.code},layer{self.result.layer:02d}]_shift_comparison.png')

  def getShiftComparisonImages(self) :
    img_orig = self.getimage(normalize=1000.,shifted=False)
    img_shifted = self.getimage(normalize=1000.,shifted=True)
    return (img_orig,img_shifted)

  @property
  def x1vec(self):
    return np.array([self.x1, self.y1])
  @property
  def x2vec(self):
    return np.array([self.x2, self.y2])

class OverlapCollection(abc.ABC):
  @abc.abstractproperty
  def overlaps(self): pass

  @property
  def overlapgraph(self):
    try:
      import networkx as nx
    except ImportError:
      raise ImportError("To get the overlap graph you have to install networkx")

    g = nx.DiGraph()
    for o in self.overlaps:
      g.add_edge(o.p1, o.p2, overlap=o)

    return g

  @property
  def overlapsdict(self):
    return {(o.p1, o.p2): o for o in self.overlaps}

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
  covxx: float = 0.
  covyy: float = 0.
  covxy: float = 0.

  @property
  def mse(self):
    return self.mse1, self.mse2, self.mse3

  @mse.setter
  def mse(self, value):
    self.mse1, self.mse2, self.mse3 = value

  @property
  def covariance(self):
    return np.array([[self.covxx, self.covxy], [self.covxy, self.covyy]])

  @covariance.setter
  def covariance(self, covariancematrix):
    assert np.isclose(covariancematrix[0, 1], covariancematrix[1, 0]), covariancematrix
    (self.covxx, self.covxy), (self.covxy, self.covyy) = covariancematrix

  @property
  def dxvec(self):
    return np.array([self.dx, self.dy])

  @property
  def dxdy(self):
    return unc.correlated_values([self.dx, self.dy], self.covariance)

  @dxdy.setter
  def dxdy(self, dxdy):
    self.dx = dxdy[0].n
    self.dy = dxdy[1].n
    self.covariance = np.array(unc.covariance_matrix(dxdy))

  @property
  def isedge(self):
    return self.tag % 2 == 0
