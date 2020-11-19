import abc, contextlib, cvxpy as cp, dataclasses, itertools, methodtools, networkx as nx, numpy as np, PIL, skimage.filters, sklearn.linear_model, typing, uncertainties as unc

from ..alignment.computeshift import computeshift
from ..alignment.overlap import AlignmentComparison
from ..baseclasses.qptiff import QPTiff
from ..zoom.zoom import ZoomSample
from ..utilities import units
from ..utilities.misc import covariance_matrix, dataclass_dc_init, floattoint
from ..utilities.tableio import readtable, writetable
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

class AnnoWarpSample(ZoomSample):
  def __init__(self, *args, bigtilepixels=(1400, 2100), tilepixels=100, tilebrightnessthreshold=45, mintilebrightfraction=0.2, mintilerange=45, **kwargs):
    super().__init__(*args, **kwargs)
    self.wsilayer = 1
    self.qptifflayer = 1
    self.__bigtilepixels = np.array(bigtilepixels)
    self.__tilepixels = tilepixels
    self.tilebrightnessthreshold = tilebrightnessthreshold
    self.mintilebrightfraction = mintilebrightfraction
    self.mintilerange = mintilerange

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
  def tilesize(self): return units.Distance(pixels=self.__tilepixels, pscale=self.imscale)
  @property
  def bigtilesize(self): return units.distances(pixels=self.__bigtilepixels, pscale=self.imscale)
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

    results = AnnoWarpAlignmentResults()
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
        imageshandle=self.getimages,
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
          AnnoWarpAlignmentResult(
            **alignmentresultkwargs,
            dxvec=(
              units.Distance(pixels=unc.ufloat(0, 9999.), pscale=self.imscale),
              units.Distance(pixels=unc.ufloat(0, 9999.), pscale=self.imscale),
            ),
            exit=255,
            exception=e,
          )
        )
      else:
        results.append(
          AnnoWarpAlignmentResult(
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
  def alignmentcsv(self): return self.csv(f"warp-{self.__tilepixels}")

  def writealignments(self, *, filename=None):
    if filename is None: filename = self.alignmentcsv
    writetable(filename, self.__alignmentresults, logger=self.logger)

  def readalignments(self, *, filename=None):
    if filename is None: filename = self.alignmentcsv
    results = self.__alignmentresults = AnnoWarpAlignmentResults(readtable(filename, AnnoWarpAlignmentResult, extrakwargs={"pscale": self.imscale, "tilesize": self.tilesize, "bigtilesize": self.bigtilesize, "imageshandle": self.getimages}))
    return results

  @property
  def logmodule(self):
    return "annowarp"

  def stitch_cvxpy(self):
    coeffrelativetobigtile = cp.Variable(shape=(2, 2))
    bigtileindexcoeff = cp.Variable(shape=(2, 2))
    constant = cp.Variable(shape=2)

    tominimize = 0
    onepixel = units.Distance(pixels=1, pscale=self.imscale)
    for result in self.__alignmentresults.goodconnectedresults:
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

    self.__stitchresult = AnnoWarpStitchResultDefaultModelCvxpy(
      problem=prob,
      coeffrelativetobigtile=coeffrelativetobigtile,
      bigtileindexcoeff=bigtileindexcoeff,
      constant=constant,
      imscale=self.imscale,
    )

    return self.__stitchresult

  @property
  def stitchcsv(self): return self.csv(f"warp-{self.__tilepixels}-stitch")

  def writestitchresult(self, *, filename=None):
    if filename is None: filename = self.stitchcsv
    self.__stitchresult.writestitchresult(filename=filename, logger=self.logger)

class AnnoWarpStitchResultBase(abc.ABC):
  def __init__(self, *, imscale, **kwargs):
    self.imscale = imscale
    super().__init__(**kwargs)

  @abc.abstractmethod
  def dxvec(self, alignmentresult):
    pass

  def residual(self, alignmentresult):
    return alignmentresult.dxvec - self.dxvec(alignmentresult)

  def writestitchresult(self, *, filename, **kwargs):
    writetable(filename, self.stitchresultentries, **kwargs)

  @abc.abstractproperty
  def stitchresultentries(self): pass

class AnnoWarpStitchResultCvxpyBase(AnnoWarpStitchResultBase):
  def __init__(self, *, problem, **kwargs):
    self.problem = problem
    super().__init__(**kwargs)

  def residual(self, alignmentresult):
    return units.nominal_values(super().residual(alignmentresult))

class AnnoWarpStitchResultDefaultModelBase(AnnoWarpStitchResultBase):
  def __init__(self, *, coeffrelativetobigtile, bigtileindexcoeff, constant, **kwargs):
    self.coeffrelativetobigtile = coeffrelativetobigtile
    self.bigtileindexcoeff = bigtileindexcoeff
    self.constant = constant
    super().__init__(**kwargs)

  def dxvec(self, alignmentresult):
    return (
      self.coeffrelativetobigtile @ alignmentresult.centerrelativetobigtile
      + self.bigtileindexcoeff @ alignmentresult.bigtileindex
      + self.constant
    )

  @property
  def stitchresultentries(self):
    return (
      AnnoWarpStitchResultEntry(
        n=1,
        value=self.coeffrelativetobigtile[0,0],
        description="coefficient of delta x as a function of x within the tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=2,
        value=self.coeffrelativetobigtile[0,1],
        description="coefficient of delta x as a function of y within the tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=3,
        value=self.coeffrelativetobigtile[1,0],
        description="coefficient of delta y as a function of x within the tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=4,
        value=self.coeffrelativetobigtile[1,1],
        description="coefficient of delta y as a function of x within the tile",
        pscale=self.imscale,
      ),

      AnnoWarpStitchResultEntry(
        n=5,
        value=self.bigtileindexcoeff[0,0],
        description="coefficient of delta x as a function of tile index in x",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=6,
        value=self.bigtileindexcoeff[0,1],
        description="coefficient of delta x as a function of tile index in y",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=7,
        value=self.bigtileindexcoeff[1,0],
        description="coefficient of delta y as a function of tile index in x",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=8,
        value=self.bigtileindexcoeff[1,1],
        description="coefficient of delta y as a function of tile index in x",
        pscale=self.imscale,
      ),

      AnnoWarpStitchResultEntry(
        n=9,
        value=self.constant[0],
        description="constant piece in delta x",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=10,
        value=self.constant[1],
        description="constant piece in delta y",
        pscale=self.imscale,
      ),
    )

class AnnoWarpStitchResultDefaultModelCvxpy(AnnoWarpStitchResultDefaultModelBase, AnnoWarpStitchResultCvxpyBase):
  def __init__(self, *, coeffrelativetobigtile, bigtileindexcoeff, constant, imscale, **kwargs):
    onepixel = units.Distance(pixels=1, pscale=imscale)
    super().__init__(
      coeffrelativetobigtile=coeffrelativetobigtile.value,
      bigtileindexcoeff=bigtileindexcoeff.value * onepixel,
      constant=constant.value * onepixel,
      imscale=imscale,
      **kwargs,
    )
    self.coeffrelativetobigtilevar = coeffrelativetobigtile
    self.bigtileindexcoeffvar = bigtileindexcoeff
    self.constantvar = constant

@dataclasses.dataclass
class AnnoWarpStitchResultEntry(DataClassWithDistances):
  pixelsormicrons = "pixels"
  def __powerfordescription(self):
    return {
      "coefficient of delta x as a function of x within the tile": 0,
      "coefficient of delta x as a function of y within the tile": 0,
      "coefficient of delta y as a function of x within the tile": 0,
      "coefficient of delta y as a function of y within the tile": 0,
      "coefficient of delta x as a function of tile index in x": 1,
      "coefficient of delta x as a function of tile index in y": 1,
      "coefficient of delta y as a function of tile index in x": 1,
      "coefficient of delta y as a function of tile index in y": 1,
      "constant piece in delta x": 1,
      "constant piece in delta y": 1,
    }[self.description]
  n: int
  value: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=__powerfordescription)
  description: str
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False

@dataclass_dc_init
class AnnoWarpAlignmentResult(AlignmentComparison, DataClassWithDistances):
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
  imageshandle: dataclasses.InitVar[typing.Callable[[], typing.Tuple[np.ndarray, np.ndarray]]] = None
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

    self.use_gpu = False
    return self.__dc_init__(*args, **kwargs)

  def __post_init__(self, tilesize, bigtilesize, exception=None, imageshandle=None, *args, **kwargs):
    super().__post_init__(*args, **kwargs)
    self.tilesize = tilesize
    self.bigtilesize = bigtilesize
    self.exception = exception
    self.imageshandle = imageshandle

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
  def tileindex(self):
    return self.xvec // self.tilesize
  @property
  def bigtileindex(self):
    return self.center // self.bigtilesize
  @property
  def bigtilecorner(self):
    return self.bigtileindex * self.bigtilesize
  @property
  def centerrelativetobigtile(self):
    return self.center - self.bigtilecorner

  @property
  def unshifted(self):
    wsi, qptiff = self.imageshandle()
    wsitile = wsi[
      units.pixels(self.y, pscale=self.pscale):units.pixels(self.y+self.tilesize, pscale=self.pscale),
      units.pixels(self.x, pscale=self.pscale):units.pixels(self.x+self.tilesize, pscale=self.pscale),
    ]
    qptifftile = qptiff[
      units.pixels(self.y, pscale=self.pscale):units.pixels(self.y+self.tilesize, pscale=self.pscale),
      units.pixels(self.x, pscale=self.pscale):units.pixels(self.x+self.tilesize, pscale=self.pscale),
    ]
    return wsitile, qptifftile

class AnnoWarpAlignmentResults(list):
  @property
  def goodresults(self):
    return type(self)(r for r in self if not r.exit)
  @methodtools.lru_cache()
  @property
  def tilesize(self):
    result, = {_.tilesize for _ in self}
    return result
  @methodtools.lru_cache()
  @property
  def pscale(self):
    result, = {_.pscale for _ in self}
    return result
  @property
  def adjacencygraph(self):
    g = nx.Graph()
    dct = {tuple(_.tileindex): _ for _ in self}

    for (ix, iy), tile in dct.items():
      g.add_node(tile.n, alignmentresult=tile, idx=(ix, iy))
      for otheridx in (ix+1, iy), (ix-1, iy), (ix, iy+1), (ix, iy-1):
        if otheridx in dct:
          g.add_edge(tile.n, dct[otheridx].n)

    return g

  @property
  def goodconnectedresults(self):
    onepixel = units.Distance(pixels=1, pscale=self.pscale)
    good = self.goodresults
    g = good.adjacencygraph
    tiledict = {tile.n: tile for tile in self}
    keep = {}
    for island in nx.connected_components(g):
      if len(island) <= 7:
        for n in island: keep[n] = False
        continue
      tiles = [tiledict[n] for n in island]
      x = np.array([t.x for t in tiles])
      y = np.array([t.y for t in tiles])
      if len(set(x)) == 1 or len(set(y)) == 1:
        for n in island: keep[n] = False
        continue
      X = np.array([x, y]).T / onepixel
      dx = np.array([t.dx for t in tiles]) / onepixel
      wx = 1/np.array([t.covxx for t in tiles]) * onepixel**2
      dy = np.array([t.dy for t in tiles]) / onepixel
      wy = 1/np.array([t.covyy for t in tiles]) * onepixel**2
      dxfit = sklearn.linear_model.LinearRegression()
      dxfit.fit(X, dx, sample_weight=wx)
      dyfit = sklearn.linear_model.LinearRegression()
      dyfit.fit(X, dy, sample_weight=wy)
      for t in tiles:
        if abs(dxfit.predict(np.array([[t.x, t.y]])/onepixel)*onepixel - t.dx) > 10*onepixel or abs(dyfit.predict(np.array([[t.x, t.y]])/onepixel)*onepixel - t.dy) > 10*onepixel:
          keep[t.n] = False
        else:
          keep[t.n] = True
    return type(self)(_ for _ in good if keep[_.n])
