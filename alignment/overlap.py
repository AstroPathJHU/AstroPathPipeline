import abc, dataclasses, matplotlib.pyplot as plt, networkx as nx, numpy as np, typing, uncertainties as unc

from .computeshift import computeshift, mse, shiftimg
from ..prepdb.rectangle import rectangleoroverlapfilter as overlapfilter
from ..prepdb.overlap import Overlap
from ..utilities import units
from ..utilities.misc import covariance_matrix, dataclass_dc_init, floattoint
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

@dataclasses.dataclass
class AlignmentOverlap(Overlap):
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
    hh, ww = units.distances(pixels=[hh, ww], pscale=self.pscale)

    image1x1 = self.x1
    image1y1 = self.y1
    image2x1 = self.x2
    image2y1 = self.y2
    image1x2 = image1x1 + ww
    image2x2 = image2x1 + ww
    image1y2 = image1y1 + hh
    image2y2 = image2y1 + hh

    overlapx1 = max(image1x1, image2x1)
    overlapx2 = min(image1x2, image2x2)
    overlapy1 = max(image1y1, image2y1)
    overlapy2 = min(image1y2, image2y2)

    offsetimage1x1 = units.Distance(pscale=self.pscale, pixels=int(units.pixels(overlapx1 - image1x1, pscale=self.pscale)))
    offsetimage1x2 = units.Distance(pscale=self.pscale, pixels=int(units.pixels(overlapx2 - image1x1, pscale=self.pscale)))
    offsetimage1y1 = units.Distance(pscale=self.pscale, pixels=int(units.pixels(overlapy1 - image1y1, pscale=self.pscale)))
    offsetimage1y2 = units.Distance(pscale=self.pscale, pixels=int(units.pixels(overlapy2 - image1y1, pscale=self.pscale)))

    offsetimage2x1 = units.Distance(pscale=self.pscale, pixels=int(units.pixels(overlapx1 - image2x1, pscale=self.pscale)))
    offsetimage2x2 = units.Distance(pscale=self.pscale, pixels=int(units.pixels(overlapx2 - image2x1, pscale=self.pscale)))
    offsetimage2y1 = units.Distance(pscale=self.pscale, pixels=int(units.pixels(overlapy1 - image2y1, pscale=self.pscale)))
    offsetimage2y2 = units.Distance(pscale=self.pscale, pixels=int(units.pixels(overlapy2 - image2y1, pscale=self.pscale)))

    cutimage1x1 = floattoint(units.pixels(offsetimage1x1, pscale=self.pscale)) + self.nclip
    cutimage1x2 = floattoint(units.pixels(offsetimage1x2, pscale=self.pscale)) - self.nclip
    cutimage1y1 = floattoint(units.pixels(offsetimage1y1, pscale=self.pscale)) + self.nclip
    cutimage1y2 = floattoint(units.pixels(offsetimage1y2, pscale=self.pscale)) - self.nclip

    cutimage2x1 = floattoint(units.pixels(offsetimage2x1, pscale=self.pscale)) + self.nclip
    cutimage2x2 = floattoint(units.pixels(offsetimage2x2, pscale=self.pscale)) - self.nclip
    cutimage2y1 = floattoint(units.pixels(offsetimage2y1, pscale=self.pscale)) + self.nclip
    cutimage2y2 = floattoint(units.pixels(offsetimage2y2, pscale=self.pscale)) - self.nclip

    #make sure that even with floattoint() they're the same size
    deltax = min(cutimage1x2 - cutimage1x1, cutimage2x2 - cutimage2x1)
    cutimage1x2 = cutimage1x1 + deltax
    cutimage2x2 = cutimage2x1 + deltax
    deltay = min(cutimage1y2 - cutimage1y1, cutimage2y2 - cutimage2y1)
    cutimage1y2 = cutimage1y1 + deltay
    cutimage2y2 = cutimage2y1 + deltay

    #positioncutimage1 = np.array([image1x1 + offsetimage1x1, image1y1 + offsetimage1y1])
    #positioncutimage2 = np.array([image2x1 + offsetimage2x1, image2y1 + offsetimage2y1])

    return (
      image1[cutimage1y1:cutimage1y2,cutimage1x1:cutimage1x2],
      image2[cutimage2y1:cutimage2y2,cutimage2x1:cutimage2x2],
    )

  def align(self, *, debug=False, alreadyalignedstrategy="error", **computeshiftkwargs):
    if self.result is None:
      alreadyalignedstrategy = None
    else:
      if alreadyalignedstrategy == "error":
        raise RuntimeError(f"Overlap {self.n} is already aligned.  To keep the previous result, call align(alreadyalignedstrategy='skip').  To align again and overwrite the previous result, call align(alreadyalignedstrategy='overwrite').")
      elif alreadyalignedstrategy == "skip":
        return self.result
      elif alreadyalignedstrategy == "overwrite":
        pass
      elif alreadyalignedstrategy == "shift_only":
        kwargs1 = {k: getattr(self.result, k) for k in ("dxvec", "exit")}
      else:
        raise ValueError(f"Unknown value alreadyalignedstrategy={alreadyalignedstrategy!r}")

    try:
      if alreadyalignedstrategy != "shift_only":
        kwargs1 = self.__computeshift(**computeshiftkwargs)
      kwargs2 = self.__shiftclip(dxvec=kwargs1["dxvec"])
      self.result = AlignmentResult(
        **self.alignmentresultkwargs,
        **kwargs1,
        **kwargs2,
      )
    except Exception as e:
      if debug: raise
      self.result = AlignmentResult(
        exit=3,
        dxvec=(units.Distance(pixels=unc.ufloat(0, 9999), pscale=self.pscale), units.Distance(pixels=unc.ufloat(0, 9999), pscale=self.pscale)),
        sc=1.,
        mse=(0., 0., 0.),
        exception=e,
        **self.alignmentresultkwargs,
      )
    return self.result

  @property
  def alignmentresultkwargs(self):
    return dict(
      n=self.n,
      p1=self.p1,
      p2=self.p2,
      code=self.tag,
      layer=self.layer,
      pscale=self.pscale,
    )

  def getinversealignment(self, inverse):
    assert (inverse.p1, inverse.p2) == (self.p2, self.p1)
    self.result = AlignmentResult(
      exit = inverse.result.exit,
      dxvec = -inverse.result.dxvec,
      sc = 1/inverse.result.sc,
      mse1 = inverse.result.mse2,
      mse2 = inverse.result.mse1,
      mse3 = inverse.result.mse3 / inverse.result.sc**2,
      **self.alignmentresultkwargs,
    )
    return self.result

  def __computeshift(self, **computeshiftkwargs):
    minimizeresult = computeshift(self.cutimages, **computeshiftkwargs)
    return {
      "dxvec": units.correlated_distances(
        pixels=(minimizeresult.dx, minimizeresult.dy),
        pscale=self.pscale,
        power=1,
      ),
      "exit": minimizeresult.exit,
    }

  @property
  def shifted(self):
    return shiftimg(self.cutimages, *units.nominal_values(units.pixels(self.result.dxvec)))

  def __shiftclip(self, dxvec):
    """
    Shift images symetrically by fractional amount
    and save the result. Compute the mse and the
    illumination correction
    """
    b1, b2 = shiftimg(self.cutimages, *units.nominal_values(units.pixels(dxvec)))

    mse1 = mse(b1)
    mse2 = mse(b2)

    sc = (mse1 / mse2) ** 0.5

    diff = b1 - b2*sc

    return {
      "sc": sc,
      "mse": (mse1, mse2, mse(diff))
    }

  def getimage(self,normalize=100.,shifted=True) :
    if shifted:
      red, green = self.shifted
    else:
      red, green = self.cutimages
    blue = (red+green)/2
    img = np.array([red, green, blue]).transpose(1, 2, 0) / normalize
    return img

  def showimages(self, normalize=100., shifted=True, saveas=None, ticks=False, **savekwargs):
    img=self.getimage(normalize,shifted)
    plt.imshow(img)
    if ticks:
      plt.xlabel("$x$")
      plt.ylabel("$y$")
    else:
      plt.xticks([])
      plt.yticks([])  # to hide tick values on X and Y axis
    if saveas is None:
      plt.show()
    else:
      plt.savefig(saveas, **savekwargs)
      plt.close()

  def getShiftComparisonImageCodeNameTuple(self) :
    return (self.result.code,f'overlap_{self.result.n}_[{self.result.p1}x{self.result.p2},type{self.result.code},layer{self.result.layer:02d}]_shift_comparison.png')

  def getShiftComparisonImages(self) :
    img_orig = self.getimage(normalize=1000.,shifted=False)
    img_shifted = self.getimage(normalize=1000.,shifted=True)
    return (img_orig,img_shifted)

@dataclass_dc_init(frozen=True)
class AlignmentResult(DataClassWithDistances):
  pixelsormicrons = "pixels"

  n: int
  p1: int
  p2: int
  code: int
  layer: int
  exit: int
  dx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  dy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  sc: float
  mse1: float
  mse2: float
  mse3: float
  covxx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  covyy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  covxy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  pscale: dataclasses.InitVar[float] = None
  exception: typing.Optional[Exception] = dataclasses.field(default=None, metadata={"includeintable": False})
  readingfromfile: dataclasses.InitVar[bool] = False

  def __init__(self, *args, **kwargs):
    dxvec = kwargs.pop("dxvec", None)
    if dxvec is not None:
      kwargs["dx"] = dxvec[0].n
      kwargs["dy"] = dxvec[1].n
      kwargs["covariance"] = covariance_matrix(dxvec)

    covariancematrix = kwargs.pop("covariance", None)
    if covariancematrix is not None:
      units.np.testing.assert_allclose(covariancematrix[0, 1], covariancematrix[1, 0])
      (kwargs["covxx"], kwargs["covxy"]), (kwargs["covxy"], kwargs["covyy"]) = covariancematrix

    mse = kwargs.pop("mse", None)
    if mse is not None:
      kwargs["mse1"], kwargs["mse2"], kwargs["mse3"] = mse

    return self.__dc_init__(*args, **kwargs)

  @property
  def mse(self):
    return self.mse1, self.mse2, self.mse3

  @property
  def covariance(self):
    return np.array([[self.covxx, self.covxy], [self.covxy, self.covyy]])

  @property
  def dxvec(self):
    return np.array(units.correlated_distances(distances=[self.dx, self.dy], covariance=self.covariance))

  @property
  def isedge(self):
    return self.tag % 2 == 0
