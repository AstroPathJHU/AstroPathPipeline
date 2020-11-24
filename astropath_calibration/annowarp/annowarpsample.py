import abc, contextlib, cvxpy as cp, dataclasses, itertools, methodtools, networkx as nx, numpy as np, PIL, skimage.filters, sklearn.linear_model, typing, uncertainties as unc

from ..alignment.computeshift import computeshift
from ..alignment.overlap import AlignmentComparison
from ..baseclasses.csvclasses import Vertex
from ..baseclasses.qptiff import QPTiff
from ..zoom.zoom import ZoomSample
from ..utilities import units
from ..utilities.misc import covariance_matrix, dataclass_dc_init, floattoint
from ..utilities.tableio import readtable, writetable
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield
from .stitch import AnnoWarpStitchResultDefaultModel, AnnoWarpStitchResultDefaultModelCvxpy, ThingWithImscale

class AnnoWarpSample(ZoomSample, ThingWithImscale):
  def __init__(self, *args, bigtilepixels=(1400, 2100), bigtileoffsetpixels=(0, 1000), tilepixels=100, tilebrightnessthreshold=45, mintilebrightfraction=0.2, mintilerange=45, **kwargs):
    super().__init__(*args, **kwargs)
    self.wsilayer = 1
    self.qptifflayer = 1
    self.__bigtilepixels = np.array(bigtilepixels)
    self.__bigtileoffsetpixels = np.array(bigtileoffsetpixels)
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
  def bigtileoffset(self): return units.distances(pixels=self.__bigtileoffsetpixels, pscale=self.imscale)
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
  def apscale(self): return self.__imageinfo["apscale"]
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
    bigtileoffset = self.bigtileoffset
    #deltax = self.deltax
    #deltay = self.deltay

    onepixel = self.oneimpixel

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
      if n%100==0 or n==ntiles: self.logger.debug("aligning tile %d/%d", n, ntiles)
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
        bigtileoffset=bigtileoffset,
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
    results = self.__alignmentresults = AnnoWarpAlignmentResults(readtable(filename, AnnoWarpAlignmentResult, extrakwargs={"pscale": self.imscale, "tilesize": self.tilesize, "bigtilesize": self.bigtilesize, "bigtileoffset": self.bigtileoffset, "imageshandle": self.getimages}))
    return results

  @property
  def logmodule(self):
    return "annowarp"

  @staticmethod
  def stitchresultcls(*, model, cvxpy):
    return {
      "default": (AnnoWarpStitchResultDefaultModel, AnnoWarpStitchResultDefaultModelCvxpy),
    }[model][cvxpy]

  def stitch(self, *, model="default"):
    stitchresultcls = self.stitchresultcls(model=model, cvxpy=False)
    nparams = stitchresultcls.nparams()
    A = np.zeros(shape=(nparams, nparams), dtype=units.unitdtype)
    b = np.zeros(shape=nparams, dtype=units.unitdtype)
    c = 0

    for result in self.__alignmentresults.goodconnectedresults:
      addA, addb, addc = stitchresultcls.Abccontributions(result)
      A += addA
      b += addb
      c += addc

    result = units.np.linalg.solve(2*A, -b)

    delta2nllfor1sigma = 1
    covariancematrix = units.np.linalg.inv(A) * delta2nllfor1sigma
    result = np.array(units.correlated_distances(distances=result, covariance=covariancematrix))

    self.__stitchresult = stitchresultcls(result, A=A, b=b, c=c, imscale=self.imscale)
    return self.__stitchresult

  def stitch_cvxpy(self, *, model="default"):
    stitchresultcls = self.stitchresultcls(model=model, cvxpy=True)
    variables = stitchresultcls.makecvxpyvariables()

    tominimize = 0
    onepixel = self.oneimpixel
    for result in self.__alignmentresults.goodconnectedresults:
      residual = stitchresultcls.cvxpyresidual(result, **variables)
      tominimize += cp.quad_form(residual, units.np.linalg.inv(result.covariance) * onepixel**2)

    minimize = cp.Minimize(tominimize)
    prob = cp.Problem(minimize)
    prob.solve()

    self.__stitchresult = stitchresultcls(
      problem=prob,
      imscale=self.imscale,
      **variables,
    )

    return self.__stitchresult

  @property
  def stitchcsv(self): return self.csv(f"warp-{self.__tilepixels}-stitch")

  def writestitchresult(self, *, filename=None):
    if filename is None: filename = self.stitchcsv
    self.__stitchresult.writestitchresult(filename=filename, logger=self.logger)

  @property
  def oldverticescsv(self): return self.csv("vertices")
  @property
  def newverticescsv(self): return self.csv("vertices-warped")

  @methodtools.lru_cache()
  @property
  def vertices(self, *, filename=None):
    if filename is None: filename = self.oldverticescsv
    return readtable(filename, QPTiffVertex, extrakwargs={"pscale": self.imscale, "bigtilesize": self.bigtilesize, "bigtileoffset": self.bigtileoffset})

  @methodtools.lru_cache()
  @property
  def warpedvertices(self):
    return [
      QPTiffVertex(
        xvec=v.xvec + units.nominal_values(self.__stitchresult.dxvec(v)),
        regionid=v.regionid,
        vid=v.vid,
        pscale=v.pscale,
        bigtilesize=v.bigtilesize,
        bigtileoffset=v.bigtileoffset,
      ) for v in self.vertices
    ]

  def writevertices(self, *, filename=None):
    if filename is None: filename = self.newverticescsv
    writetable(filename, self.warpedvertices)

class QPTiffCoordinateBase(abc.ABC):
  @abc.abstractproperty
  def bigtilesize(self): pass
  @abc.abstractproperty
  def bigtileoffset(self): pass
  @abc.abstractproperty
  def qptiffcoordinate(self): pass
  @property
  def bigtileindex(self):
    return (self.xvec - self.bigtileoffset) // self.bigtilesize
  @property
  def bigtilecorner(self):
    return self.bigtileindex * self.bigtilesize + self.bigtileoffset
  @property
  def centerrelativetobigtile(self):
    return self.qptiffcoordinate - self.bigtilecorner

class QPTiffCoordinate(QPTiffCoordinateBase):
  def __init__(self, *args, bigtilesize, bigtileoffset, **kwargs):
    self.__bigtilesize = bigtilesize
    self.__bigtileoffset = bigtileoffset
    super().__init__(*args, **kwargs)
  @property
  def bigtilesize(self): return self.__bigtilesize
  @property
  def bigtileoffset(self): return self.__bigtileoffset

class QPTiffVertex(QPTiffCoordinate, Vertex):
  @property
  def qptiffcoordinate(self):
    return self.xvec

@dataclass_dc_init
class AnnoWarpAlignmentResult(AlignmentComparison, QPTiffCoordinateBase, DataClassWithDistances):
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
  bigtileoffset: dataclasses.InitVar[units.Distance]
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
    self.__dc_init__(*args, **kwargs)

  def __post_init__(self, tilesize, bigtilesize, bigtileoffset, exception=None, imageshandle=None, *args, **kwargs):
    super().__post_init__(*args, **kwargs)
    self.tilesize = tilesize
    self.__bigtilesize = bigtilesize
    self.__bigtileoffset = bigtileoffset
    self.exception = exception
    self.imageshandle = imageshandle

  @property
  def bigtilesize(self): return self.__bigtilesize
  @property
  def bigtileoffset(self): return self.__bigtileoffset

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
  qptiffcoordinate = center
  @property
  def tileindex(self):
    return self.xvec // self.tilesize

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

class AnnoWarpAlignmentResults(list, units.ThingWithPscale):
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
    onepixel = self.onepixel
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
