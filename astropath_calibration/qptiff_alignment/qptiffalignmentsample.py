import contextlib, cvxpy as cp, dataclasses, itertools, methodtools, numpy as np, PIL, skimage.filters, uncertainties as unc

from ..alignment.computeshift import computeshift
from ..alignment.overlap import AlignmentComparison
from ..baseclasses.qptiff import QPTiff
from ..zoom.zoom import ZoomSample
from ..utilities import units
from ..utilities.misc import covariance_matrix, dataclass_dc_init, floattoint
from ..utilities.tableio import readtable, writetable
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

class QPTiffAlignmentSample(ZoomSample):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.wsilayer = 1
    self.qptifflayer = 1
    self.__bigtilesize = np.array([1400, 2100])
    self.__tilesize = 100
    self.tilebrightnessthreshold = 45
    self.mintilebrightfraction = 0.2
    self.mintilerange = 45

    self.__nentered = 0
    self.__using_images_context = self.enter_context(contextlib.ExitStack())

    self.__images = None

  @contextlib.contextmanager
  def using_images(self):
    if self.__nentered == 0:
      self.__using_images_context.enter_context(self.PILmaximagepixels())
      self.__wsi = self.__using_images_context.enter_context(PIL.Image.open(self.wsifilename(layer=self.wsilayer)))
      self.__qptiff = self.__using_images_context.enter_context(QPTiff(self.qptifffilename))
    self.__nentered += 1
    try:
      if self.__nentered == 1:
        self.__imageinfo #access it now to make sure it's cached
      yield self.__wsi, self.__qptiff
    finally:
      self.__nentered -= 1
      if self.__nentered == 0:
        self.__wsi = self.__qptiff = None
        self.__using_images_context.close()

  @property
  def tilesize(self): return units.Distance(pixels=self.__tilesize, pscale=self.imscale)
  @property
  def bigtilesize(self): return units.distances(pixels=self.__bigtilesize, pscale=self.imscale)
  @property
  def deltax(self): return units.Distance(pixels=self.__deltax, pscale=self.imscale)
  @property
  def deltay(self): return units.Distance(pixels=self.__deltay, pscale=self.imscale)

  @methodtools.lru_cache()
  @property
  def __imageinfo(self):
    with self.using_images() as (wsi, fqptiff):
      zoomlevel = fqptiff.zoomlevels[0]
      apscale = zoomlevel.qpscale
      ipscale = self.pscale / apscale
      ppscale = floattoint(np.round(float(ipscale)))
      iqscale = ipscale / ppscale
      imscales = {apscale * iqscale, self.pscale / ppscale}
      imscale, = imscales
      return {
        "apscale": apscale,
        "ipscale": ipscale,
        "ppscale": ppscale,
        "iqscale": iqscale,
        "imscale": imscale,
        "xposition": fqptiff.xposition,
        "yposition": fqptiff.yposition,
      }

  @property
  def ppscale(self): return self.__imageinfo["ppscale"]
  @property
  def iqscale(self): return self.__imageinfo["iqscale"]
  @property
  def imscale(self): return self.__imageinfo["imscale"]
  @property
  def xposition(self): return self.__imageinfo["xposition"]
  @property
  def yposition(self): return self.__imageinfo["yposition"]

  def getimages(self, keep=False):
    if self.__images is not None: return self.__images
    with self.using_images() as (wsi, fqptiff):
      zoomlevel = fqptiff.zoomlevels[0]
      qptiff = PIL.Image.fromarray(zoomlevel[self.qptifflayer-1].asarray())

      wsisize = np.array(wsi.size, dtype=np.uint)
      qptiffsize = np.array(qptiff.size, dtype=np.uint)
      wsisize //= self.ppscale
      qptiffsize = (qptiffsize * self.iqscale).astype(np.uint)
      wsi = wsi.resize(wsisize)
      qptiff = qptiff.resize(qptiffsize)

      newsize = 0, 0, np.min((wsisize[0], qptiffsize[0])), np.min((wsisize[1], qptiffsize[1]))
      wsi = wsi.crop(newsize)
      qptiff = qptiff.crop(newsize)

      wsi = np.asarray(wsi)
      qptiff = np.asarray(qptiff)

      if keep: self.__images = wsi, qptiff
      return wsi, qptiff

  def align(self, *, debug=False, write_result=False):
    wsi, qptiff = self.getimages()
    imscale = self.imscale
    tilesize = self.tilesize
    bigtilesize = self.bigtilesize
    #deltax = self.deltax
    #deltay = self.deltay

    onepixel = units.Distance(pixels=1, pscale=imscale)

    mx1 = units.convertpscale(min(field.mx1 for field in self.rectangles), self.pscale, imscale, 1)
    mx2 = units.convertpscale(max(field.mx2 for field in self.rectangles), self.pscale, imscale, 1)
    my1 = units.convertpscale(min(field.my1 for field in self.rectangles), self.pscale, imscale, 1)
    my2 = units.convertpscale(max(field.my2 for field in self.rectangles), self.pscale, imscale, 1)

    #nx1 = max(floattoint(mx1 // deltax), 1)
    #nx2 = floattoint(mx2 // deltax) + 1
    #ny1 = max(1, floattoint(my1 // deltay), 1)
    #ny2 = floattoint(my2 // deltay) + 1

    #ex = np.arange(nx1, nx2+1) * self.deltax
    #ey = np.arange(ny1, ny2+1) * self.deltay

    #tweak the y position by -900 for the microsocope glitches
    #(from Alex's code.  I don't know what this means.)
    qshifty = 0
    if self.yposition == 0: qshifty = units.Distance(pixels=900, pscale=imscale)

    mx2 = min(mx2, units.Distance(pixels=wsi.shape[1], pscale=imscale) - tilesize)
    my2 = min(my2, units.Distance(pixels=wsi.shape[0], pscale=imscale) - tilesize)

    n1 = floattoint(my1//tilesize)-1
    n2 = floattoint(my2//tilesize)+1
    m1 = floattoint(mx1//tilesize)-1
    m2 = floattoint(mx2//tilesize)+1

    results = []
    ntiles = (m2+1-m1) * (n2+1-n1)
    self.logger.info("aligning %d tiles", ntiles)
    for n, (ix, iy) in enumerate(itertools.product(np.arange(m1, m2+1), np.arange(n1, n2+1)), start=1):
      self.logger.debug("aligning tile %d/%d", n, ntiles)
      x = tilesize * (ix-1)
      y = tilesize * (iy-1)
      if y+onepixel-qshifty <= 0: continue
      wsitile = wsi[
        units.pixels(y, pscale=imscale):units.pixels(y+tilesize, pscale=imscale),
        units.pixels(x, pscale=imscale):units.pixels(x+tilesize, pscale=imscale),
      ]
      brightfraction = np.mean(wsitile>self.tilebrightnessthreshold)
      if brightfraction < self.mintilebrightfraction: continue
      if np.max(wsitile) - np.min(wsitile) < self.mintilerange: continue
      qptifftile = qptiff[
        units.pixels(y, pscale=imscale):units.pixels(y+tilesize, pscale=imscale),
        units.pixels(x, pscale=imscale):units.pixels(x+tilesize, pscale=imscale),
      ]

      alignmentresultkwargs = dict(
        n=n,
        x=x,
        y=y,
        mi=brightfraction,
        pscale=imscale,
        tilesize=tilesize,
        bigtilesize=bigtilesize,
      )

      wsitile = wsitile - skimage.filters.gaussian(wsitile, sigma=20)
      wsitile = skimage.filters.gaussian(wsitile, sigma=3)
      qptifftile = qptifftile - skimage.filters.gaussian(qptifftile, sigma=20)
      qptifftile = skimage.filters.gaussian(qptifftile, sigma=3)
      try:
        shiftresult = computeshift((wsitile, qptifftile), usemaxmovementcut=False)
      except Exception as e:
        if debug: raise
        results.append(
          QPTiffAlignmentResult(
            **alignmentresultkwargs,
            dxvec=(
              units.Distance(pixels=unc.ufloat(0, 9999.), pscale=self.imscale),
              units.Distance(pixels=unc.ufloat(0, 9999.), pscale=self.imscale),
            ),
            exit=255,
          )
        )
      else:
        results.append(
          QPTiffAlignmentResult(
            **alignmentresultkwargs,
            dxvec=units.correlated_distances(
              pixels=(shiftresult.dx, shiftresult.dy),
              pscale=imscale,
              power=1,
            ),
            exit=shiftresult.exit,
          )
        )
    self.__alignmentresults = results
    if write_result:
      self.writealignments()
    return results

  @property
  def alignmentcsv(self): return self.csv(f"warp-{self.__tilesize}")

  def writealignments(self, *, filename=None):
    if filename is None: filename = self.alignmentcsv
    writetable(filename, self.__alignmentresults, logger=self.logger)

  def readalignments(self, *, filename=None):
    if filename is None: filename = self.alignmentcsv
    results = self.__alignmentresults = readtable(filename, QPTiffAlignmentResult, extrakwargs={"pscale": self.imscale, "tilesize": self.tilesize, "bigtilesize": self.bigtilesize})
    return results

  @property
  def logmodule(self):
    return "annowarp"

  def plotresult(self, result, **kwargs):
    wsi, qptiff = self.getimages()
    AlignedTile(result, wsi, qptiff).showimages(**kwargs)

  def stitch_cvxpy(self):
    coeffrelativetobigtile = cp.Variable(shape=(2, 2))
    bigtileindexcoeff = cp.Variable(shape=(2, 2))
    constant = cp.Variable(shape=2)

    tominimize = 0
    onepixel = units.Distance(pixels=1, pscale=self.imscale)
    for result in self.__alignmentresults:
      if result.exit: continue
      residual = (
        units.nominal_values(result.dxvec)/onepixel - (
          coeffrelativetobigtile @ (result.centerrelativetobigtile/onepixel)
          + bigtileindexcoeff @ result.bigtileindex
          + constant
        )
      )
      tominimize += cp.quad_form(residual, units.np.linalg.inv(result.covariance) * onepixel**2)

    minimize = cp.Minimize(tominimize)
    prob = cp.Problem(minimize)
    prob.solve()

    return prob, tominimize, coeffrelativetobigtile, bigtileindexcoeff, constant

@dataclass_dc_init(frozen=True)
class QPTiffAlignmentResult(DataClassWithDistances):
  pixelsormicrons = "pixels"
  n: int
  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  dx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  dy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  covxx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  covxy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  covyy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  mi: float
  exit: int
  tilesize: dataclasses.InitVar[units.Distance]
  bigtilesize: dataclasses.InitVar[units.Distance]
  exceptions: dataclasses.InitVar[Exception] = None
  pscale: dataclasses.InitVar[float] = None
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

    return self.__dc_init__(*args, **kwargs)

  def __post_init__(self, tilesize, bigtilesize, exception=None, *args, **kwargs):
    super().__post_init__(*args, **kwargs)
    object.__setattr__(self, "tilesize", tilesize)
    object.__setattr__(self, "bigtilesize", bigtilesize)
    object.__setattr__(self, "exception", exception)

  @property
  def xvec(self):
    return np.array([self.x, self.y])
  @property
  def covariance(self):
    return np.array([[self.covxx, self.covxy], [self.covxy, self.covyy]])
  @property
  def dxvec(self):
    return np.array(units.correlated_distances(distances=[self.dx, self.dy], covariance=self.covariance))
  @property
  def center(self):
    return self.xvec + self.tilesize/2
  @property
  def bigtileindex(self):
    return self.center // self.bigtilesize
  @property
  def bigtilecorner(self):
    return self.bigtileindex * self.bigtilesize
  @property
  def centerrelativetobigtile(self):
    return self.center - self.bigtilecorner

class AlignedTile(AlignmentComparison):
  def __init__(self, result, wsi, qptiff, *args, **kwargs):
    self.__result = result
    self.__wsi = wsi
    self.__qptiff = qptiff
    self.__imscale = result.pscale
    self.__tilesize = result.tilesize
    super().__init__(*args, **kwargs)
  @property
  def pscale(self): return self.__imscale
  @property
  def unshifted(self):
    wsitile = self.__wsi[
      units.pixels(self.__result.y, pscale=self.__imscale):units.pixels(self.__result.y+self.__tilesize, pscale=self.__imscale),
      units.pixels(self.__result.x, pscale=self.__imscale):units.pixels(self.__result.x+self.__tilesize, pscale=self.__imscale),
    ]
    qptifftile = self.__qptiff[
      units.pixels(self.__result.y, pscale=self.__imscale):units.pixels(self.__result.y+self.__tilesize, pscale=self.__imscale),
      units.pixels(self.__result.x, pscale=self.__imscale):units.pixels(self.__result.x+self.__tilesize, pscale=self.__imscale),
    ]
    return wsitile, qptifftile
  @property
  def dxvec(self): return self.__result.dxvec
