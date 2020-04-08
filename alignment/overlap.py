import abc, dataclasses, matplotlib.pyplot as plt, networkx as nx, numpy as np, typing, uncertainties as unc

from .computeshift import computeshift, mse, shiftimg
from .rectangle import rectangleoroverlapfilter as overlapfilter
from ..utilities import units
from ..utilities.misc import covariance_matrix, dataclass_dc_init

@dataclasses.dataclass
class Overlap:
  n: int
  p1: int
  p2: int
  x1: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: x.microns, "readfunction": float})
  y1: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: x.microns, "readfunction": float})
  x2: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: x.microns, "readfunction": float})
  y2: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: x.microns, "readfunction": float})
  tag: int

  def setalignmentinfo(self, *, layer, pscale, nclip, rectangles):
    self.layer = layer
    self.pscale = pscale
    self.nclip = nclip
    self.rectangles = rectangles
    self.result = None

    self.x1 = units.Distance(microns=self.x1, pscale=self.pscale)
    self.y1 = units.Distance(microns=self.y1, pscale=self.pscale)
    self.x2 = units.Distance(microns=self.x2, pscale=self.pscale)
    self.y2 = units.Distance(microns=self.y2, pscale=self.pscale)

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
    image1x1 = int(self.x1.pixels)
    image1y1 = int(self.y1.pixels)
    image2x1 = int(self.x2.pixels)
    image2y1 = int(self.y2.pixels)
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
    try:
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
        dxvec=(units.udistance(pixels=unc.ufloat(0, 9999), pscale=self.pscale), units.udistance(pixels=unc.ufloat(0, 9999), pscale=self.pscale)),
        sc=1.,
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
    return shiftimg(self.cutimages, self.result.dx.pixels, self.result.dy.pixels)

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

  @property
  def x1vec(self):
    return np.array([self.x1, self.y1])
  @property
  def x2vec(self):
    return np.array([self.x2, self.y2])

class OverlapCollection(abc.ABC):
  @abc.abstractproperty
  def overlaps(self): pass

  def overlapgraph(self, useexitstatus=False):
    g = nx.DiGraph()
    for o in self.overlaps:
      if useexitstatus and o.result.exit: continue
      g.add_edge(o.p1, o.p2, overlap=o)

    return g

  def nislands(self, *args, **kwargs):
    return nx.number_strongly_connected_components(self.overlapgraph(*args, **kwargs))

  @property
  def overlapsdict(self):
    return {(o.p1, o.p2): o for o in self.overlaps}

  @property
  def overlaprectangleindices(self):
    return frozenset(o.p1 for o in self.overlaps) | frozenset(o.p2 for o in self.overlaps)

  @property
  def selectoverlaprectangles(self):
    return overlapfilter(self.overlaprectangleindices)

class OverlapList(list, OverlapCollection):
  @property
  def overlaps(self): return self

@dataclass_dc_init(frozen=True)
class AlignmentResult:
  n: int
  p1: int
  p2: int
  code: int
  layer: int
  exit: int
  dx: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: units.pixels(x), "readfunction": float})
  dy: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: units.pixels(x), "readfunction": float})
  sc: float
  mse1: float
  mse2: float
  mse3: float
  covxx: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: units.pixels(x), "readfunction": float})
  covyy: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: units.pixels(x), "readfunction": float})
  covxy: units.Distance = dataclasses.field(metadata={"writefunction": lambda x: units.pixels(x), "readfunction": float})
  pscale: dataclasses.InitVar[float] = None
  exception: typing.Optional[Exception] = dataclasses.field(default=None, metadata={"includeintable": False})

  def __init__(self, *args, **kwargs):
    dxvec = kwargs.pop("dxvec", None)
    if dxvec is not None:
      kwargs["dx"] = dxvec[0].n
      kwargs["dy"] = dxvec[1].n
      kwargs["covariance"] = covariance_matrix(dxvec)

    covariancematrix = kwargs.pop("covariance", None)
    if covariancematrix is not None:
      assert np.isclose(units.pixels(covariancematrix)[0, 1], units.pixels(covariancematrix)[1, 0]), covariancematrix
      (kwargs["covxx"], kwargs["covxy"]), (kwargs["covxy"], kwargs["covyy"]) = covariancematrix

    mse = kwargs.pop("mse", None)
    if mse is not None:
      kwargs["mse1"], kwargs["mse2"], kwargs["mse3"] = mse

    return self.__dc_init__(*args, **kwargs)

  def __post_init__(self, pscale):
    pscale = {pscale} if pscale is not None else set()
    pscale |= {_.pscale for _ in (self.dx, self.dy, self.covxx, self.covxy, self.covyy) if isinstance(_, units.Distance) and _.pscale is not None}
    if not pscale:
      raise TypeError("Have to either provide pscale explicitly or give coordinates in units.Distance form")
    if len(pscale) > 1:
      raise units.UnitsError("Provided inconsistent pscales")
    pscale = pscale.pop()

    if not isinstance(self.dx, units.Distance):
      super().__setattr__("dx", units.Distance(pixels=self.dx, pscale=pscale))
    if not isinstance(self.dy, units.Distance):
      super().__setattr__("dy", units.Distance(pixels=self.dy, pscale=pscale))
    if not isinstance(self.covxx, units.Distance):
      super().__setattr__("covxx", units.Distance(pixels=self.covxx, pscale=pscale, power=2))
    if not isinstance(self.covxy, units.Distance):
      super().__setattr__("covxy", units.Distance(pixels=self.covxy, pscale=pscale, power=2))
    if not isinstance(self.covyy, units.Distance):
      super().__setattr__("covyy", units.Distance(pixels=self.covyy, pscale=pscale, power=2))

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
