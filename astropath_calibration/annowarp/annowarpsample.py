import abc, contextlib, csv, cvxpy as cp, itertools, methodtools, more_itertools, networkx as nx, numpy as np, PIL, skimage.filters, sklearn.linear_model, uncertainties as unc

from ..alignment.computeshift import computeshift
from ..alignment.overlap import AlignmentComparison
from ..baseclasses.csvclasses import Region, Vertex
from ..baseclasses.polygon import Polygon
from ..baseclasses.qptiff import QPTiff
from ..zoom.zoom import ZoomSample
from ..utilities import units
from ..utilities.dataclasses import MyDataClass
from ..utilities.misc import covariance_matrix, floattoint
from ..utilities.tableio import readtable, writetable
from ..utilities.units.dataclasses import DataClassWithPscale, distancefield
from .stitch import AnnoWarpStitchResultDefaultModel, AnnoWarpStitchResultDefaultModelCvxpy, ThingWithImscale

class AnnoWarpSample(ZoomSample, ThingWithImscale):
  def __init__(self, *args, bigtilepixels=(1400, 2100), bigtileoffsetpixels=(0, 1000), tilepixels=100, tilebrightnessthreshold=45, mintilebrightfraction=0.2, mintilerange=45, **kwargs):
    super().__init__(*args, **kwargs)
    self.wsilayer = 1
    self.qptifflayer = 1
    self.__bigtilepixels = np.array(bigtilepixels)
    self.__bigtileoffsetpixels = np.array(bigtileoffsetpixels)
    self.__tilepixels = tilepixels
    if np.any(self.__bigtilepixels % self.__tilepixels) or np.any(self.__bigtileoffsetpixels % self.__tilepixels):
      raise ValueError("You should set the tilepixels {self.__tilepixels} so that it divides bigtilepixels {self.__bigtilepixels} and bigtileoffset {self.__bigtileoffsetpixels}")
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

    self.logger.info("doing the initial rough alignment")
    #first align the two with respect to each other
    #in case there are big shifts, this makes sure the tiles line up
    #and can be aligned
    zoomfactor = 5
    wsizoom = PIL.Image.fromarray(wsi)
    wsizoom = np.asarray(wsizoom.resize(np.array(wsizoom.size)//zoomfactor))
    qptiffzoom = PIL.Image.fromarray(qptiff)
    qptiffzoom = np.asarray(qptiffzoom.resize(np.array(qptiffzoom.size)//zoomfactor))
    firstresult = computeshift((qptiffzoom, wsizoom), usemaxmovementcut=False)

    initialdx = floattoint(np.rint(firstresult.dx.n * zoomfactor / self.__tilepixels) * self.__tilepixels)
    initialdy = floattoint(np.rint(firstresult.dy.n * zoomfactor / self.__tilepixels) * self.__tilepixels)

    if initialdx or initialdy:
      self.logger.warning(f"found a relative shift of around {initialdx, initialdy} pixels between the qptiff and wsi")

    wsix1 = wsiy1 = qptiffx1 = qptiffy1 = 0
    qptiffy2, qptiffx2 = qptiff.shape
    wsiy2, wsix2 = wsi.shape
    if initialdx > 0:
      wsix1 += initialdx
      qptiffx2 -= initialdx
    else:
      qptiffx1 -= initialdx
      wsix2 += initialdx
    if initialdy > 0:
      wsiy1 += initialdy
      qptiffy2 -= initialdy
    else:
      qptiffy1 -= initialdy
      wsiy2 += initialdy

    wsi = wsi[wsiy1:wsiy2, wsix1:wsix2]
    qptiff = qptiff[qptiffy1:qptiffy2, qptiffx1:qptiffx2]

    onepixel = self.oneimpixel

    imscale = self.imscale
    tilesize = self.tilesize
    bigtilesize = self.bigtilesize
    bigtileoffset = self.bigtileoffset

    mx1 = units.convertpscale(min(field.mx1 for field in self.rectangles), self.pscale, imscale, 1)
    mx2 = units.convertpscale(max(field.mx2 for field in self.rectangles), self.pscale, imscale, 1)
    my1 = units.convertpscale(min(field.my1 for field in self.rectangles), self.pscale, imscale, 1)
    my2 = units.convertpscale(max(field.my2 for field in self.rectangles), self.pscale, imscale, 1)

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
        floattoint(units.pixels(y, pscale=imscale)):floattoint(units.pixels(y+tilesize, pscale=imscale)),
        floattoint(units.pixels(x, pscale=imscale)):floattoint(units.pixels(x+tilesize, pscale=imscale)),
      ]
      if not wsitile.size: continue
      brightfraction = np.mean(wsitile>self.tilebrightnessthreshold)
      if brightfraction < self.mintilebrightfraction: continue
      if np.max(wsitile) - np.min(wsitile) < self.mintilerange: continue
      qptifftile = qptiff[
        units.pixels(y, pscale=imscale):units.pixels(y+tilesize, pscale=imscale),
        units.pixels(x, pscale=imscale):units.pixels(x+tilesize, pscale=imscale),
      ]

      alignmentresultkwargs = dict(
        n=n,
        x=x+qptiffx1,
        y=y+qptiffy1,
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
        shiftresult = computeshift((qptifftile, wsitile), usemaxmovementcut=False)
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
              pixels=(shiftresult.dx+initialdx, shiftresult.dy+initialdy),
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

  def stitch(self, *args, cvxpy=False, **kwargs):
    return (self.stitch_cvxpy if cvxpy else self.stitch_nocvxpy)(*args, **kwargs)

  def stitch_nocvxpy(self, **kwargs):
    allkwargs = kwargs.copy()
    model = kwargs.pop("model", "default")
    constraintmus = kwargs.pop("constraintmus", None)
    constraintsigmas = kwargs.pop("constraintsigmas", None)
    residualpullcutoff = kwargs.pop("residualpullcutoff", 5)
    floatedparams = kwargs.pop("floatedparams", "all")
    _removetiles = kwargs.pop("_removetiles", [])
    _choosetiles = kwargs.pop("_choosetiles", "bigislands")
    if kwargs: raise TypeError(f"Unknown kwargs {kwargs}")

    if constraintmus is constraintsigmas is None:
      self.logger.info("doing the global fit")
    else:
      self.logger.warningglobal("doing the global fit with constraints")
    stitchresultcls = self.stitchresultcls(model=model, cvxpy=False)
    nparams = stitchresultcls.nparams()
    A = np.zeros(shape=(nparams, nparams), dtype=units.unitdtype)
    b = np.zeros(shape=nparams, dtype=units.unitdtype)
    c = 0

    alignmentresults = AnnoWarpAlignmentResults(_ for _ in self.__alignmentresults if _.n not in _removetiles)
    if _choosetiles == "bigislands":
      alignmentresults = alignmentresults.goodconnectedresults(minislandsize=8)
      if len(alignmentresults) < 15:
        self.logger.warning("didn't find good alignment results in big islands, trying to stitch with smaller islands")
        allkwargs["_choosetiles"] = "smallislands"
        return self.stitch_nocvxpy(**allkwargs)
    elif _choosetiles == "smallislands":
      alignmentresults = alignmentresults.goodconnectedresults(minislandsize=4)
      if len(alignmentresults) < 15:
        self.logger.warning("didn't find good alignment results in small islands, using all good alignment results for stitching")
        allkwargs["_choosetiles"] = "all"
        return self.stitch_nocvxpy(**allkwargs)
    elif _choosetiles == "all":
      alignmentresults = alignmentresults.goodresults
    else:
      raise ValueError(f"Invalid _choosetiles {_choosetiles}")

    A, b, c = stitchresultcls.Abc(alignmentresults, constraintmus, constraintsigmas, floatedparams=floatedparams)

    try:
      result = units.np.linalg.solve(2*A, -b)
    except np.linalg.LinAlgError:
      if _choosetiles == "bigislands":
        self.logger.warning("fit failed using big islands, trying to stitch with smaller islands")
        allkwargs["_choosetiles"] = "smallislands"
        return self.stitch_nocvxpy(**allkwargs)
      if _choosetiles == "smallislands":
        self.logger.warning("fit failed using small islands, using all good alignment results for stitching")
        allkwargs["_choosetiles"] = "all"
        return self.stitch_nocvxpy(**allkwargs)
      raise

    delta2nllfor1sigma = 1
    covariancematrix = units.np.linalg.inv(A) * delta2nllfor1sigma
    result = np.array(units.correlated_distances(distances=result, covariance=covariancematrix))

    stitchresult = stitchresultcls(result, A=A, b=b, c=c, imscale=self.imscale)

    if residualpullcutoff is not None:
      removemoretiles = []
      debuglines = []
      for result in alignmentresults:
        residualsq = np.sum(stitchresult.residual(result, apscale=self.imscale)**2)
        if abs(residualsq.n / residualsq.s) > residualpullcutoff:
          removemoretiles.append(result.n)
          debuglines.append(f"{result.n} {residualsq}")
      if removemoretiles:
        self.logger.warning(f"Alignment results {removemoretiles} are outliers (> {residualpullcutoff} sigma residuals), removing them and trying again")
        for l in debuglines: self.logger.debug(l)
        allkwargs["_removetiles"]=_removetiles+removemoretiles
        return self.stitch(**allkwargs)

    self.__stitchresult = stitchresult
    return self.__stitchresult

  def stitch_cvxpy(self, **kwargs):
    allkwargs = kwargs.copy()
    model = kwargs.pop("model", "default")
    constraintmus = kwargs.pop("constraintmus", None)
    constraintsigmas = kwargs.pop("constraintsigmas", None)
    _removetiles = kwargs.pop("_removetiles", [])
    _choosetiles = kwargs.pop("_choosetiles", "bigislands")
    if kwargs: raise TypeError(f"Unknown kwargs {kwargs}")

    stitchresultcls = self.stitchresultcls(model=model, cvxpy=True)
    variables = stitchresultcls.makecvxpyvariables()

    alignmentresults = AnnoWarpAlignmentResults(_ for _ in self.__alignmentresults if _.n not in _removetiles)
    if _choosetiles == "bigislands":
      alignmentresults = alignmentresults.goodconnectedresults(minislandsize=8)
      if len(alignmentresults) < 15:
        self.logger.warning("didn't find good alignment results in big islands, trying to stitch with smaller islands")
        allkwargs["_choosetiles"] = "smallislands"
        return self.stitch_nocvxpy(**allkwargs)
    elif _choosetiles == "smallislands":
      alignmentresults = alignmentresults.goodconnectedresults(minislandsize=4)
      if len(alignmentresults) < 15:
        self.logger.warning("didn't find good alignment results in small islands, using all good alignment results for stitching")
        allkwargs["_choosetiles"] = "all"
        return self.stitch_nocvxpy(**allkwargs)
    elif _choosetiles == "all":
      alignmentresults = alignmentresults.goodresults
    else:
      raise ValueError(f"Invalid _choosetiles {_choosetiles}")

    tominimize = 0
    onepixel = self.oneimpixel
    for result in alignmentresults:
      residual = stitchresultcls.cvxpyresidual(result, **variables)
      tominimize += cp.quad_form(residual, units.np.linalg.inv(result.covariance) * onepixel**2)

    tominimize += stitchresultcls.constraintquadforms(variables, constraintmus, constraintsigmas, imscale=self.imscale)

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
  @property
  def oldregionscsv(self): return self.csv("regions")
  @property
  def newregionscsv(self): return self.csv("regions-warped")

  @methodtools.lru_cache()
  def __getvertices(self, *, apscale, filename=None):
    if filename is None: filename = self.oldverticescsv
    extrakwargs={
     "apscale": apscale,
     "bigtilesize": units.convertpscale(self.bigtilesize, self.imscale, apscale),
     "bigtileoffset": units.convertpscale(self.bigtileoffset, self.imscale, apscale)
    }
    with open(filename) as f:
      reader = csv.DictReader(f)
      if "wx" in reader.fieldnames and "wy" in reader.fieldnames:
        typ = WarpedVertex
      else:
        typ = QPTiffVertex
    vertices = readtable(filename, typ, extrakwargs=extrakwargs)
    if typ == WarpedVertex:
      vertices = [v.originalvertex for v in vertices]
    return vertices

  @property
  def vertices(self):
    return self.__getvertices(apscale=self.apscale)

  @methodtools.lru_cache()
  def __getwarpedvertices(self, *, apscale):
    oneapmicron = units.onemicron(pscale=apscale)
    onemicron = self.onemicron
    onepixel = self.onepixel
    return [
      WarpedVertex(
        vertex=v,
        wxvec=(v.xvec + units.nominal_values(self.__stitchresult.dxvec(v, apscale=apscale))) / oneapmicron * onemicron // onepixel * onepixel,
        pscale=self.pscale,
      ) for v in self.__getvertices(apscale=apscale)
    ]

  @property
  def warpedvertices(self):
    return self.__getwarpedvertices(apscale=self.apscale)

  @methodtools.lru_cache()
  def __getregions(self, *, apscale, filename=None):
    if filename is None: filename = self.oldregionscsv
    return readtable(filename, Region, extrakwargs={"apscale": apscale, "pscale": self.pscale})

  @property
  def regions(self):
    return self.__getregions(apscale=self.apscale)

  @methodtools.lru_cache()
  @property
  def warpedregions(self):
    regions = self.regions
    warpedverticesiterator = iter(self.warpedvertices)
    result = []
    for i, region in enumerate(regions, start=1):
      zipfunction = more_itertools.zip_equal if i == len(regions) else zip
      newvertices = []
      polyvertices = [v for v in self.vertices if v.regionid == region.regionid]
      for oldvertex, newvertex in zipfunction(polyvertices, warpedverticesiterator):
        np.testing.assert_array_equal(
          np.round((oldvertex.xvec / oldvertex.oneappixel).astype(float)),
          np.round((newvertex.xvec / oldvertex.oneappixel).astype(float)),
        )
        newvertices.append(newvertex.finalvertex)
      result.append(
        Region(
          regionid=region.regionid,
          sampleid=region.sampleid,
          layer=region.layer,
          rid=region.rid,
          isNeg=region.isNeg,
          type=region.type,
          nvert=region.nvert,
          pscale=region.pscale,
          apscale=region.apscale,
          poly=Polygon(vertices=[newvertices], pscale=region.pscale, apscale=region.apscale)
        ),
      )
    return result

  def writevertices(self, *, filename=None):
    self.logger.info("writing vertices")
    if filename is None: filename = self.newverticescsv
    writetable(filename, self.warpedvertices)

  def writeregions(self, *, filename=None):
    self.logger.info("writing regions")
    if filename is None: filename = self.newregionscsv
    writetable(filename, self.warpedregions)

  def runannowarp(self, *, readalignments=False, model="default", constraintmus=None, constraintsigmas=None):
    if not readalignments:
      self.align()
      self.writealignments()
    else:
      self.readalignments()
    self.stitch(model=model, constraintmus=constraintmus, constraintsigmas=constraintsigmas)
    self.writestitchresult()
    self.writevertices()
    self.writeregions()

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

class QPTiffCoordinate(MyDataClass, QPTiffCoordinateBase):
  def __user_init__(self, *args, bigtilesize, bigtileoffset, **kwargs):
    self.__bigtilesize = bigtilesize
    self.__bigtileoffset = bigtileoffset
    super().__user_init__(*args, **kwargs)
  @property
  def bigtilesize(self): return self.__bigtilesize
  @property
  def bigtileoffset(self): return self.__bigtileoffset

class QPTiffVertex(QPTiffCoordinate, Vertex):
  @classmethod
  def transforminitargs(cls, *args, vertex=None, **kwargs):
    vertexkwargs = {"vertex": vertex}
    if isinstance(vertex, QPTiffVertex):
      vertexkwargs.update({
        "bigtilesize": vertex.bigtilesize,
        "bigtileoffset": vertex.bigtileoffset,
      })
    return super().transforminitargs(
      *args,
      **kwargs,
      **vertexkwargs,
    )
  @property
  def qptiffcoordinate(self):
    return self.xvec

class WarpedVertex(QPTiffVertex):
  __pixelsormicrons = "pixels"
  wx: distancefield(pixelsormicrons=__pixelsormicrons, dtype=int)
  wy: distancefield(pixelsormicrons=__pixelsormicrons, dtype=int)

  @classmethod
  def transforminitargs(cls, *args, wxvec=None, **kwargs):
    wxveckwargs = {}
    if wxvec is not None:
      wxveckwargs["wx"], wxveckwargs["wy"] = wxvec

    return super().transforminitargs(
      *args,
      **kwargs,
      **wxveckwargs,
    )

  @property
  def wxvec(self): return np.array([self.wx, self.wy])

  @property
  def originalvertex(self):
    return QPTiffVertex(
      regionid=self.regionid,
      vid=self.vid,
      xvec=self.xvec,
      apscale=self.apscale,
      pscale=self.pscale,
      bigtilesize=self.bigtilesize,
      bigtileoffset=self.bigtileoffset,
    )

  @property
  def finalvertex(self):
    return Vertex(
      regionid=self.regionid,
      vid=self.vid,
      im3xvec=self.wxvec,
      apscale=self.apscale,
      pscale=self.pscale,
    )

class AnnoWarpAlignmentResult(AlignmentComparison, QPTiffCoordinateBase, DataClassWithPscale):
  pixelsormicrons = "pixels"
  n: int
  x: distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  y: distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  dx: distancefield(pixelsormicrons=pixelsormicrons)
  dy: distancefield(pixelsormicrons=pixelsormicrons)
  covxx: distancefield(pixelsormicrons=pixelsormicrons, power=2)
  covxy: distancefield(pixelsormicrons=pixelsormicrons, power=2)
  covyy: distancefield(pixelsormicrons=pixelsormicrons, power=2)
  mi: float
  exit: int

  @classmethod
  def transforminitargs(cls, *args, **kwargs):
    dxvec = kwargs.pop("dxvec", None)
    if dxvec is not None:
      kwargs["dx"] = dxvec[0].n
      kwargs["dy"] = dxvec[1].n
      kwargs["covariance"] = covariance_matrix(dxvec)

    covariancematrix = kwargs.pop("covariance", None)
    if covariancematrix is not None:
      units.np.testing.assert_allclose(covariancematrix[0, 1], covariancematrix[1, 0])
      (kwargs["covxx"], kwargs["covxy"]), (kwargs["covxy"], kwargs["covyy"]) = covariancematrix

    return super().transforminitargs(*args, **kwargs)

  def __user_init__(self, tilesize, bigtilesize, bigtileoffset, exception=None, imageshandle=None, *args, **kwargs):
    self.use_gpu = False
    super().__user_init__(*args, **kwargs)
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

  def goodconnectedresults(self, *, minislandsize=8):
    onepixel = self.onepixel
    good = self.goodresults
    g = good.adjacencygraph
    tiledict = {tile.n: tile for tile in self}
    keep = {}
    for island in nx.connected_components(g):
      if len(island) < minislandsize:
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
