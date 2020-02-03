import dataclasses, matplotlib.pyplot as plt, numpy as np

from .computeshift import computeshift, mse, shiftimg

@dataclasses.dataclass(eq=False, repr=False)
class Overlap:
  n: int
  p1: int
  p2: int
  x1: float
  y1: float
  x2: float
  y2: float
  tag: int

  def setalignmentinfo(self, layer, pscale, nclip, images):
    self.layer = layer
    self.pscale = pscale
    self.nclip = nclip
    self.images = images[(self.p1-1,self.p2-1),:,:]

  def align(self):
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

    return self.result

  def getinversealignment(self, inverse):
    assert (inverse.p1, inverse.p2) == (self.p2, self.p1)
    self.cutimages = inverse.cutimages[::-1,...]
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
      dv = inverse.result.dv,
      R_error_stat = inverse.result.R_error_stat,
      R_error_syst = inverse.result.R_error_syst,
      F_error = inverse.result.F_error,
    )
    self.result.covariance = inverse.result.covariance
    self.result.minimizeresult = inverse.result.minimizeresult
    self.shifted = np.array([
      inverse.shifted[1],
      inverse.shifted[0],
      inverse.shifted[2],
    ])
    return self.result

  def prepimage(self):
    hh, ww = self.images.shape[1:]

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

    self.cutimages = np.array([
      self.images[0,cutimage1y1:cutimage1y2,cutimage1x1:cutimage1x2],
      self.images[1,cutimage2y1:cutimage2y2,cutimage2x1:cutimage2x2],
    ])

  def computeshift(self):
    minimizeresult = computeshift(self.cutimages)
    self.result.minimizeresult = minimizeresult
    self.result.dx = minimizeresult.dx
    self.result.dy = minimizeresult.dy
    self.result.dv = minimizeresult.dv
    self.result.R_error_stat = minimizeresult.R_error_stat
    self.result.R_error_syst = minimizeresult.R_error_syst
    self.result.F_error = minimizeresult.F_error
    self.result.covariance = minimizeresult.covariance

  def shiftclip(self):
    """
    Shift images symetrically by fractional amount
    and save the result. Compute the mse and the
    illumination correction
    """
    self.shifted = A = shiftimg(self.cutimages,self.result.dx,self.result.dy)

    #clip the non-overlapping parts
    ww = 10*(1+int(max(np.abs([self.result.dx, self.result.dy]))/10))

    b1, b2, average = self.result.overlapregion = A[:, ww:-ww or None, ww:-ww or None]

    mse1 = mse(b1)
    mse2 = mse(b2)

    self.result.sc = (mse1 / mse2) ** 0.5

    diff = b1 - b2*self.result.sc
    self.result.mse = mse1, mse2, mse(diff)

  def showimages(self, normalize=100., shifted=True):
    if shifted:
      red, blue, _ = self.shifted
    else:
      red, blue = self.cutimages

    green = np.zeros(red.shape)

    img = np.array([red, green, blue]).transpose(1, 2, 0) / normalize

    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


@dataclasses.dataclass(eq=False)
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
  R_error_stat: float = 0.
  R_error_syst: float = 0.
  F_error: float = 0.
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
