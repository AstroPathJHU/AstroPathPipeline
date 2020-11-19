import dataclasses, matplotlib.pyplot as plt, methodtools, more_itertools, numpy as np, typing, uncertainties as unc

from .computeshift import computeshift, mse, shiftimg
from ..baseclasses.overlap import Overlap
from ..utilities import units
from ..utilities.misc import covariance_matrix, dataclass_dc_init, floattoint
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

@dataclasses.dataclass
class AlignmentOverlap(Overlap):
  def __init__(self, *args, layer1=None, layer2=None, **kwargs):
    self.__use_gpu = False
    super().__init__(*args, **kwargs)
    if layer1 is None:
      try:
        self.rectangles[0].layer
      except AttributeError:
        raise ValueError(f"Have to tell the overlap which layer you're using for rectangle 1. choices: {self.rectangles[0].layers}")
    if layer2 is None:
      try:
        self.rectangles[1].layer
      except AttributeError:
        raise ValueError(f"Have to tell the overlap which layer you're using for rectangle 1. choices: {self.rectangles[1].layers}")
    self.__layers = layer1, layer2

  def __hash__(self):
    if not self.ismultilayer: return super().__hash__()
    return hash((super().__hash__(), self.layers))
  def __eq__(self, other):
    return super().__eq__(other) and self.layers == other.layers

  @property
  def layers(self): return self.__layers
  @property
  def layer1(self): return self.__layers[0]
  @property
  def layer2(self): return self.__layers[1]
  @methodtools.lru_cache()
  @property
  def ismultilayer(self):
    return any(_ is not None for _ in self.layers)

  @property
  def images(self):
    images = [None, None]
    with self.rectangles[0].using_image() as images[0], self.rectangles[1].using_image() as images[1]:
      result = tuple(image[:, :] if layer is None else image[r.layers.index(layer), :, :] for r, image, layer in more_itertools.zip_equal(self.rectangles, images, self.layers))
      for i in result: i.flags.writeable = False
      return result

  @methodtools.lru_cache()
  @property
  def cutimageslices(self):
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

    onepixel = self.onepixel

    offsetimage1x1 = (overlapx1 - image1x1) // onepixel * onepixel
    offsetimage1x2 = (overlapx2 - image1x1) // onepixel * onepixel
    offsetimage1y1 = (overlapy1 - image1y1) // onepixel * onepixel
    offsetimage1y2 = (overlapy2 - image1y1) // onepixel * onepixel

    offsetimage2x1 = (overlapx1 - image2x1) // onepixel * onepixel
    offsetimage2x2 = (overlapx2 - image2x1) // onepixel * onepixel
    offsetimage2y1 = (overlapy1 - image2y1) // onepixel * onepixel
    offsetimage2y2 = (overlapy2 - image2y1) // onepixel * onepixel

    cutimage1x1 = floattoint((offsetimage1x1 + self.nclip) / onepixel)
    cutimage1x2 = floattoint((offsetimage1x2 - self.nclip) / onepixel)
    cutimage1y1 = floattoint((offsetimage1y1 + self.nclip) / onepixel)
    cutimage1y2 = floattoint((offsetimage1y2 - self.nclip) / onepixel)

    cutimage2x1 = floattoint((offsetimage2x1 + self.nclip) / onepixel)
    cutimage2x2 = floattoint((offsetimage2x2 - self.nclip) / onepixel)
    cutimage2y1 = floattoint((offsetimage2y1 + self.nclip) / onepixel)
    cutimage2y2 = floattoint((offsetimage2y2 - self.nclip) / onepixel)

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
      (slice(cutimage1y1, cutimage1y2), slice(cutimage1x1, cutimage1x2)),
      (slice(cutimage2y1, cutimage2y2), slice(cutimage2x1, cutimage2x2)),
    )

  @property
  def cutimages(self):
    image1, image2 = self.images
    slice1, slice2 = self.cutimageslices
    return (
      image1[slice1],
      image2[slice2],
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
        if "gputhread" in computeshiftkwargs.keys() and "gpufftdict" in computeshiftkwargs.keys() :
          self.use_gpu = computeshiftkwargs["gputhread"] is not None and computeshiftkwargs["gpufftdict"] is not None
      kwargs2 = self.__shiftclip(dxvec=kwargs1["dxvec"])
      self.result = self.alignmentresulttype(
        **self.alignmentresultkwargs,
        **kwargs1,
        **kwargs2,
      )
    except Exception as e:
      if debug: raise
      self.result = self.alignmentresulttype(
        exit=255,
        dxvec=(units.Distance(pixels=unc.ufloat(0, 9999), pscale=self.pscale), units.Distance(pixels=unc.ufloat(0, 9999), pscale=self.pscale)),
        sc=1.,
        mse=(0., 0., 0.),
        exception=e,
        **self.alignmentresultkwargs,
      )
    return self.result

  @property
  def alignmentresulttype(self):
    if self.ismultilayer:
      return LayerAlignmentResult
    else:
      return AlignmentResult

  @property
  def alignmentresultkwargs(self):
    result = {
      "n": self.n,
      "p1": self.p1,
      "p2": self.p2,
      "code": self.tag,
      "pscale": self.pscale,
    }
    if self.ismultilayer:
      result.update({
        "layer1": self.layers[0],
        "layer2": self.layers[1],
      })
    else:
      result.update({
        "layer": self.layer,
      })
    return result

  def isinverseof(self, inverse):
    return (inverse.p1, inverse.p2) == (self.p2, self.p1) and inverse.layers == tuple(reversed(self.layers))

  def getinversealignment(self, inverse):
    assert self.isinverseof(inverse)
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
  def use_gpu(self) :
    return self.__use_gpu
  @use_gpu.setter
  def use_gpu(self,use_gpu) :
    self.__use_gpu = use_gpu

  @property
  def shifted(self):
    return shiftimg(self.cutimages, *units.nominal_values(units.pixels(self.result.dxvec)),use_gpu=self.use_gpu)

  def __shiftclip(self, dxvec):
    """
    Shift images symetrically by fractional amount
    and save the result. Compute the mse and the
    illumination correction
    """
    b1, b2 = shiftimg(self.cutimages, *units.nominal_values(units.pixels(dxvec)),use_gpu=self.use_gpu)

    mse1 = mse(b1)
    mse2 = mse(b2)

    sc = (mse1 / mse2) ** 0.5

    diff = b1 - b2*sc

    return {
      "sc": sc,
      "mse": (mse1, mse2, mse(diff))
    }

  def getimage(self,normalize=100.,shifted=True,scale=False) :
    if shifted:
      red, green = self.shifted
      if scale:
        red *= self.result.sc ** -.5
        green *= self.result.sc ** .5
    else:
      red, green = self.cutimages
    blue = (red+green)/2
    img = np.array([red, green, blue]).transpose(1, 2, 0) / normalize
    return img

  def showimages(self, normalize=100., shifted=True, scale=False, saveas=None, ticks=False, **savekwargs):
    img=self.getimage(normalize=normalize, shifted=shifted, scale=scale)
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

  def getShiftComparisonDetailTuple(self) :
    return (self.result.code,self.p1,self.p2,f'overlap_{self.n}_[{self.p1}x{self.p2},type{self.result.code},layer{self.result.layer:02d}]_shift_comparison.png')

  def getShiftComparisonImages(self) :
    img_orig = self.getimage(normalize=1000.,shifted=False)
    img_shifted = self.getimage(normalize=1000.,shifted=True)
    return (img_orig,img_shifted)

@dataclass_dc_init(frozen=True)
class AlignmentResultBase(DataClassWithDistances):
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

  @property
  def iscorner(self):
    return self.tag % 2 == 1 and self.tag != 5

  @property
  def issamerectangle(self):
    return self.tag == 5

@dataclass_dc_init(frozen=True)
class AlignmentResult(AlignmentResultBase):
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
    return super().__init__(*args, **kwargs)

@dataclass_dc_init(frozen=True)
class LayerAlignmentResult(AlignmentResultBase):
  pixelsormicrons = "pixels"

  n: int
  p1: int
  p2: int
  code: int
  layer1: int
  layer2: int
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
    return super().__init__(*args, **kwargs)
