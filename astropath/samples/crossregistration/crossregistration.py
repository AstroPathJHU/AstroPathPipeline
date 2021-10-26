import collections, contextlib, itertools, matplotlib.pyplot as plt, more_itertools, numpy as np, scipy.ndimage, skimage.registration, skimage.transform, uncertainties as unc, uncertainties.umath as umath, uncertainties.unumpy as unp

from ...shared.logging import MultiLogger
from ...slides.align.computeshift import computeshift, OptimizeResult, shiftimg
from ...slides.align.overlap import AlignmentComparison
from ...slides.annowarp.annowarpsample import WSISample
from ...slides.stitchmask.stitchmasksample import AstroPathTissueMaskSample
from ...utilities import units
from ...utilities.dataclasses import MetaDataAnnotation
from ...utilities.misc import affinetransformation, covariance_matrix, floattoint
from ...utilities.units import ThingWithPscale
from ...utilities.units.dataclasses import DataClassWithPscale, DataClassWithPscaleFrozen, distancefield, makedataclasswithpscale

class ReadWSISample(WSISample, AstroPathTissueMaskSample):
  def __init__(self, *args, uselogfiles=False, **kwargs):
    super().__init__(*args, uselogfiles=uselogfiles, **kwargs)
  @classmethod
  def logmodule(cls): return "crossregistration"
  def run(self): assert False

class ThingWithZoomedScale(ThingWithPscale, scale="zoomedscale"):
  @property
  def zoomfactor(self): return self.pscale / self.zoomedscale
DataClassWithZoomedScale, DataClassWithZoomedScaleFrozen = makedataclasswithpscale("DataClassWithZoomedScale", "zoomedscale", ThingWithZoomedScale)
class DataClassWithZoomedScale(DataClassWithZoomedScale, DataClassWithPscale): pass
class DataClassWithZoomedScaleFrozen(DataClassWithZoomedScaleFrozen, DataClassWithPscaleFrozen): pass

class CrossRegistration(contextlib.ExitStack, ThingWithZoomedScale):
  def __init__(self, *args, root1, samp1, zoomroot1, root2, samp2, zoomroot2, tilepixels=256, zoomfactor=8, mintissuefraction=0.2, dbloadroot1=None, dbloadroot2=None, logroot1=None, logroot2=None, maskroot1=None, maskroot2=None, uselogfiles=True, **kwargs):
    self.samples = (
      ReadWSISample(root=root1, samp=samp1, zoomroot=zoomroot1, dbloadroot=dbloadroot1, logroot=logroot1, maskroot=maskroot1, uselogfiles=uselogfiles),
      ReadWSISample(root=root2, samp=samp2, zoomroot=zoomroot2, dbloadroot=dbloadroot2, logroot=logroot2, maskroot=maskroot2, uselogfiles=uselogfiles),
    )
    self.__zoomfactor = zoomfactor
    super().__init__(*args, **kwargs)

    self.__nentered = collections.defaultdict(lambda: 0)
    self.__scaledwsisandmasks = {}
    self.__tilesize = tilepixels * self.onezoomedpixel
    self.__mintissuefraction = mintissuefraction
    self.__logger = MultiLogger(*(s.logger for s in self.samples), entermessage=f"cross-registering {self.SlideID1} and {self.SlideID2}")

  @property
  def SlideID1(self): return self.samples[0].SlideID
  @property
  def SlideID2(self): return self.samples[1].SlideID
  @property
  def SampleID1(self): return self.samples[0].SampleID
  @property
  def SampleID2(self): return self.samples[1].SampleID
  @property
  def REDCapID(self):
    redcapids = {s.REDCapID for s in self.samples}
    try:
      redcapid, = redcapids
    except ValueError:
      raise ValueError(f"samples {self.SlideID1, self.SlideID2} have different REDCapIDs {tuple(redcapids)}")
    return redcapid
  @property
  def pscale(self):
    pscale, = {s.pscale for s in self.samples}
    return pscale
  @property
  def zoomfactor(self): return self.__zoomfactor
  @property
  def zoomedscale(self): return self.pscale / self.zoomfactor
  @property
  def tilesize(self): return self.__tilesize
  @property
  def mintissuefraction(self): return self.__mintissuefraction
  @property
  def logger(self): return self.__logger

  @contextlib.contextmanager
  def using_wsis(self):
    with self.enter_context(contextlib.ExitStack()) as stack:
      wsis = [stack.enter_context(_.using_wsi(1)) for _ in self.samples]
      yield wsis

  @contextlib.contextmanager
  def using_tissuemasks(self):
    with self.enter_context(contextlib.ExitStack()) as stack:
      masks = [stack.enter_context(_.using_tissuemask()) for _ in self.samples]
      yield masks

  @contextlib.contextmanager
  def __using_scaled_wsis_and_masks(self, **kwargs):
    kwargskey = tuple(sorted(kwargs.items()))
    with self.using_wsis(), self.using_tissuemasks():
      if self.__nentered[kwargskey] == 0:
        self.__scaledwsisandmasks[kwargskey] = self.__getscaledwsisandmasks(**kwargs)
      try:
        yield self.__scaledwsisandmasks[kwargskey]
      finally:
        del self.__scaledwsisandmasks[kwargskey]

  def using_scaled_wsis_and_masks(self, *, smoothsigma, equalize=False):
    return self.__using_scaled_wsis_and_masks(smoothsigma=smoothsigma, equalize=equalize)

  def __getscaledwsisandmasks(self, *, smoothsigma, equalize=False):
    with self.using_wsis() as wsis, self.using_tissuemasks() as masks:
      if self.zoomfactor > 1:
        wsis = [wsi.resize(np.array(wsi.size)//self.zoomfactor) for wsi in wsis]
        masks = [skimage.transform.downscale_local_mean(mask, (self.zoomfactor, self.zoomfactor)) >= 0.5 for mask in masks]
      masks = [np.asarray(mask) for mask in masks]
      wsis = [np.asarray(wsi) for wsi in wsis]
      if smoothsigma is not None:
        wsis = [skimage.filters.gaussian(wsi, smoothsigma, mode="nearest") for wsi in wsis]
      if equalize:
        wsis = [skimage.exposure.equalize_adapthist(wsi) for wsi in wsis]
      return wsis, masks

  def runalignment(self, *, _debugprint=False, smoothsigma=10):
    with self.using_scaled_wsis_and_masks(smoothsigma=1, equalize=False) as (wsis, masks):
      wsi1, wsi2 = wsis
      mask1, mask2 = masks
      if _debugprint > .5:
        for _ in wsis+masks:
          print(_.shape, _.dtype)
          plt.imshow(_)
          plt.show()

      ysize1, xsize1 = wsi1.shape
      ygrid1, xgrid1 = np.mgrid[0:ysize1,0:xsize1].astype(np.int64)
      sumwsi1 = np.sum(wsi1)
      centroid1 = floattoint(np.rint(np.array([np.sum(wsi1*ygrid1) / sumwsi1, np.sum(wsi1*xgrid1) / sumwsi1])))
      xcumsum1 = np.cumsum(np.sum(wsi1, axis=0))
      ycumsum1 = np.cumsum(np.sum(wsi1, axis=1))
      lowersumcut1 = sumwsi1 * 0.005
      uppersumcut1 = sumwsi1 * 0.995
      xlowersize1 = centroid1[1] - np.where(xcumsum1<lowersumcut1)[0][-1]
      xuppersize1 = np.where(xcumsum1>uppersumcut1)[0][0] - centroid1[1]
      ylowersize1 = centroid1[0] - np.where(ycumsum1<lowersumcut1)[0][-1]
      yuppersize1 = np.where(ycumsum1>uppersumcut1)[0][0] - centroid1[0]

      ysize2, xsize2 = wsi2.shape
      ygrid2, xgrid2 = np.mgrid[0:ysize2,0:xsize2]
      sumwsi2 = np.sum(wsi2)
      centroid2 = floattoint(np.rint(np.array([np.sum(wsi2*ygrid2) / sumwsi2, np.sum(wsi2*xgrid2) / sumwsi2])))
      xcumsum2 = np.cumsum(np.sum(wsi2, axis=0))
      ycumsum2 = np.cumsum(np.sum(wsi2, axis=1))
      lowersumcut2 = sumwsi2 * 0.005
      uppersumcut2 = sumwsi2 * 0.995
      xlowersize2 = centroid2[1] - np.where(xcumsum2<lowersumcut2)[0][-1]
      xuppersize2 = np.where(xcumsum2>uppersumcut2)[0][0] - centroid2[1]
      ylowersize2 = centroid2[0] - np.where(ycumsum2<lowersumcut2)[0][-1]
      yuppersize2 = np.where(ycumsum2>uppersumcut2)[0][0] - centroid2[0]

      xlowersize = np.max([xlowersize1, xlowersize2], axis=0)
      xuppersize = np.max([xuppersize1, xuppersize2], axis=0)
      ylowersize = np.max([ylowersize1, ylowersize2], axis=0)
      yuppersize = np.max([yuppersize1, yuppersize2], axis=0)

      lowersize = ylowersize, xlowersize
      uppersize = yuppersize, xuppersize

      lowersize = max(lowersize)
      uppersize = max(uppersize)
      lowersize += 500
      uppersize += 500

      padlow1 = np.max([lowersize-centroid1, [0, 0]], axis=0)
      padhigh1 = np.max([centroid1 + uppersize - wsi1.shape, [0, 0]], axis=0)
      if np.any([padlow1, padhigh1]):
        wsi1 = np.pad(wsi1, ((padlow1[0], padhigh1[0]), (padlow1[1], padhigh1[1])))
        mask1 = np.pad(mask1, ((padlow1[0], padhigh1[0]), (padlow1[1], padhigh1[1])))
        centroid1 += padlow1

      padlow2 = np.max([lowersize-centroid2, [0, 0]], axis=0)
      padhigh2 = np.max([centroid2 + uppersize - wsi2.shape, [0, 0]], axis=0)
      if np.any([padlow2, padhigh2]):
        wsi2 = np.pad(wsi2, ((padlow2[0], padhigh2[0]), (padlow2[1], padhigh2[1])))
        mask2 = np.pad(mask2, ((padlow2[0], padhigh2[0]), (padlow2[1], padhigh2[1])))
        centroid2 += padlow2

      wsis = wsi1, wsi2
      masks = mask1, mask2

      slice1 = slice(
        (centroid1 - lowersize)[0],
        (centroid1 + uppersize)[0],
      ), slice(
        (centroid1 - lowersize)[1],
        (centroid1 + uppersize)[1],
      )

      slice2 = slice(
        (centroid2 - lowersize)[0],
        (centroid2 + uppersize)[0],
      ), slice(
        (centroid2 - lowersize)[1],
        (centroid2 + uppersize)[1],
      )

      firsttranslationresult = OptimizeResult(
        slice1=slice1,
        slice2=slice2,
        padlow1=padlow1,
        padhigh1=padhigh1,
        padlow2=padlow2,
        padhigh2=padhigh2,
        dx=slice1[1].start - slice2[1].start + padlow2[1] - padlow1[1],
        dy=slice1[0].start - slice2[0].start + padlow2[0] - padlow1[0],
      )
      firsttranslation = affinetransformation(translation=(firsttranslationresult.dx*self.onezoomedpixel, firsttranslationresult.dy*self.onezoomedpixel))

      wsis = wsi1, wsi2 = wsi1[slice1], wsi2[slice2]
      masks = mask1, mask2 = mask1[slice1], mask2[slice2]

      if _debugprint > .5:
        print("raw wsis")
        for _ in wsis+masks:
          print(_.shape, _.dtype)
          plt.imshow(_)
          plt.show()

      wsis = tuple(skimage.filters.gaussian(wsi, smoothsigma, mode="nearest") for wsi in wsis)

      if _debugprint > .5:
        print("smoothed")
        for _ in wsis+masks:
          print(_.shape, _.dtype)
          plt.imshow(_)
          plt.show()

      wsi1, wsi2 = wsis = tuple(skimage.exposure.equalize_adapthist(wsi) for wsi in wsis)

      if _debugprint > .5:
        print("equalized")
        for _ in wsis+masks:
          print(_.shape, _.dtype)
          plt.imshow(_)
          plt.show()

      zoommore = [skimage.transform.resize(wsi, np.asarray(wsi.shape)//8) for wsi in wsis]
      zoomevenmore = [skimage.transform.resize(wsi, np.asarray(wsi.shape)//2) for wsi in zoommore]
      r1 = self.getrotation(zoomevenmore, -180, 180-15, 15, _debugprint=_debugprint)
      r2 = self.getrotation(zoomevenmore, r1.angle.n-15, r1.angle.n+15, 2, _debugprint=_debugprint)
      r3 = self.getrotation(zoommore, r2.angle.n-2, r2.angle.n+2, 0.02, _debugprint=_debugprint)
      rotationresult = r3
      rotationresult.xcorr.update(r2.xcorr)
      rotationresult.xcorr.update(r1.xcorr)
      rotation = affinetransformation(rotation=umath.radians(rotationresult.angle))

      wsis = wsi1, wsi2 = wsi1, skimage.transform.rotate(wsi2, rotationresult.angle.n)
      mask1, mask2 = masks = mask1, skimage.transform.rotate(mask2, rotationresult.angle.n).astype(bool)

      if _debugprint > .5:
        print("rotated")
        for _ in wsis+masks:
          print(_.shape, _.dtype)
          plt.imshow(_)
          plt.show()

      translationresult = computeshift(wsis[::-1], checkpositivedefinite=False, usemaxmovementcut=False, mindistancetootherpeak=10000, showbigimage=_debugprint>0.5, showsmallimage=_debugprint>0.5)
      translation = affinetransformation(translation=(translationresult.dx*self.onezoomedpixel, translationresult.dy*self.onezoomedpixel))

      self.__initialaffinetransformation = initialaffinetransformation = translation @ rotation @ firsttranslation
      self.writecsvs("xform", [CrossRegAffineMatrix(matrix=initialaffinetransformation, pscale=self.pscale, zoomedscale=self.zoomedscale, SlideID=self.SlideID1, SlideID1=self.SlideID2, SampleID=self.SampleID1, SampleID1=self.SampleID2, REDCapID=self.REDCapID)])

      wsis = wsi1, wsi2 = tuple(shiftimg(wsis, -translationresult.dx.n, -translationresult.dy.n, shiftwhich=1))
      masks = mask1, mask2 = tuple(shiftimg(masks, -translationresult.dx.n, -translationresult.dy.n, shiftwhich=1)>0.5)

      if _debugprint > .5:
        print("shifted")
        for _ in wsis+masks:
          print(_.shape, _.dtype)
          plt.imshow(_)
          plt.show()

      masks = mask1, mask2 = tuple(self.processmask(mask) for mask in masks)

      if _debugprint > .5:
        print("processed masks")
        for _ in wsis+masks:
          print(_.shape, _.dtype)
          plt.imshow(_)
          plt.show()

      mask = mask1 | mask2
      del mask1, mask2
      masks = mask,

      if _debugprint > .5:
        print("unioned masks")
        for _ in wsis+masks:
          print(_.shape, _.dtype)
          plt.imshow(_)
          plt.show()

      shape, = {_.shape for _ in wsis+masks}
      shape = np.array(shape) * self.onezoomedpixel

      ntilesy, ntilesx = floattoint((4*np.array(shape)//self.tilesize+1).astype(float))
      ntiles = ntilesx * ntilesy
      results = []
      for n, (ix, iy) in enumerate(itertools.product(range(1, ntilesx+1), range(1, ntilesy+1)), start=1):
        if n % 100 == 0 or n == ntiles: self.logger.debug("%d / %d", n, ntiles)
        ixvec = np.array([ix, iy])
        x, y = (ixvec-1) * self.tilesize / 4
        slc = slice(
          floattoint(float(y / self.onezoomedpixel)),
          floattoint(float((y+self.tilesize) / self.onezoomedpixel)),
        ), slice(
          floattoint(float(x / self.onezoomedpixel)),
          floattoint(float((x+self.tilesize) / self.onezoomedpixel)),
        )
        maskslice = mask[slc]
        if np.count_nonzero(maskslice) / maskslice.size < self.mintissuefraction:
          continue
        tile1 = wsi1[slc]
        tile2 = wsi2[slc]
        alignmentresultkwargs = dict(
          n=n,
          x=x,
          y=y,
          pscale=self.pscale,
          zoomedscale=self.zoomedscale,
          tilesize=self.tilesize,
          initialaffinetransformation=initialaffinetransformation,
        )
        try:
          shiftresult = computeshift((tile1, tile2), usemaxmovementcut=False)
        except Exception as e:
          results.append(
            CrossRegAlignmentResult(
              **alignmentresultkwargs,
              dxvec=(
                units.Distance(pixels=unc.ufloat(0, 9999.), pscale=self.zoomedscale),
                units.Distance(pixels=unc.ufloat(0, 9999.), pscale=self.zoomedscale),
              ),
              exit=255,
              exception=e,
            )
          )
        else:
          results.append(
            CrossRegAlignmentResult(
              **alignmentresultkwargs,
              dxvec=units.correlated_distances(
                #here we apply initialdx and initialdy so that the reported
                #result is the global shift
                pixels=(shiftresult.dx, shiftresult.dy),
                pscale=self.zoomedscale,
                power=1,
              ),
              exit=shiftresult.exit,
            )
          )
      self.__alignmentresults = results
      if not results:
        raise ValueError("Couldn't align any tiles")
      self.writecsvs("xwarp", self.__alignmentresults)

      return results

  def writecsvs(self, csv, *args, **kwargs):
    for s in self.samples:
      s.writecsv(csv, *args, **kwargs)

  @staticmethod
  def getrotation(rotationwsis, minangle, maxangle, stepangle, *, _debugprint=-float("inf")):
    wsi1, wsi2 = rotationwsis
    xcorrs = {}
    angles = more_itertools.peekable(np.arange(minangle, maxangle+stepangle, stepangle))
    for angle in angles:
      rotated = wsi1, skimage.transform.rotate(wsi2, angle)
      if _debugprint > 100:
        for _ in rotated:
          print(angle)
          plt.imshow(_)
          plt.show()
      shiftresult = computeshift(rotated, checkpositivedefinite=False, usemaxmovementcut=False, mindistancetootherpeak=10000)
      xcorrs[angle] = shiftresult.crosscorrelation

      if not angles and max(xcorrs, key=xcorrs.get) == angle:
        angles.prepend(angle + stepangle)

    bestangle, bestxcorr = max(xcorrs.items(), key=lambda x: x[1].n)

    x = []
    y = []
    for angle, xcorr in xcorrs.items():
      anglediff = abs(bestangle - angle)
      if anglediff > 180:
        anglediff = abs(anglediff - 360)
      assert anglediff <= 180
      if anglediff < stepangle * 1.2:
        x.append(angle)
        y.append(xcorr)
    x = np.asarray(x)
    y = np.asarray(y)
    xmatrix = np.asarray([x**2, x, x**0]).T
    a, b, c = unp.ulinalg.inv(xmatrix) @ y
    bestangle = -b / (2*a)
    bestxcorr = a*bestangle**2 + b*bestangle + c

    return OptimizeResult(
      xcorr=xcorrs,
      angle=bestangle,
      xcorrs=bestxcorr,
      exit=0,
    )

  @staticmethod
  def processmask(mask):
    mask = scipy.ndimage.binary_fill_holes(mask)
    mask = skimage.morphology.area_opening(mask, 1000)
    disk = skimage.morphology.disk(20, dtype=np.bool)
    mask = skimage.morphology.binary_closing(mask, disk)
    mask = skimage.morphology.binary_opening(mask, disk)
    mask = skimage.morphology.binary_dilation(mask, disk)
    return mask

  @property
  def affineentry(self):
    return 

class CrossRegAlignmentResult(AlignmentComparison, DataClassWithZoomedScale):
  n: int
  x: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int, pscalename="zoomedscale")
  y: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int, pscalename="zoomedscale")
  dx: units.Distance = distancefield(pixelsormicrons="pixels", secondfunction="{:.6g}".format, pscalename="zoomedscale")
  dy: units.Distance = distancefield(pixelsormicrons="pixels", secondfunction="{:.6g}".format, pscalename="zoomedscale")
  covxx: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format, pscalename="zoomedscale")
  covxy: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format, pscalename="zoomedscale")
  covyy: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format, pscalename="zoomedscale")
  exit: int
  tilesize: units.Distance = distancefield(pixelsormicrons="pixels", includeintable=False, pscalename="zoomedscale")
  exception: Exception = MetaDataAnnotation(None, includeintable=False)
  initialaffinetransformation: np.ndarray = MetaDataAnnotation(None, includeintable=False)

  @classmethod
  def transforminitargs(cls, *args, **kwargs):
    morekwargs = {}

    dxvec = kwargs.pop("dxvec", None)
    if dxvec is not None:
      morekwargs["dx"] = dxvec[0].n
      morekwargs["dy"] = dxvec[1].n
      covariancematrix = covariance_matrix(dxvec)
    else:
      covariancematrix = kwargs.pop("covariance", None)

    if covariancematrix is not None:
      units.np.testing.assert_allclose(covariancematrix[0, 1], covariancematrix[1, 0])
      (morekwargs["covxx"], morekwargs["covxy"]), (morekwargs["covxy"], morekwargs["covyy"]) = covariancematrix

    return super().transforminitargs(*args, **kwargs, **morekwargs)

  def __post_init__(self, *args, **kwargs):
    if self.initialaffinetransformation is None:
      self.initialaffinetransformation = np.identity(3)
    super().__post_init__(*args, **kwargs)

  @property
  def covariancematrix(self):
    return np.array([[self.covxx, self.covxy], [self.covxy, self.covyy]])
  @property
  def dxvec(self): return units.correlated_distances(distances=[self.dx, self.dy], covariance=self.covariancematrix)
  @property
  def xvec(self): return np.array([self.x, self.y])
  @property
  def unshifted(self): raise NotImplementedError

  @property
  def affinetransformation(self):
    return affinetransformation(translation=self.dxvec) @ self.initialaffinetransformation

class CrossRegAffineMatrix(DataClassWithZoomedScale):
  REDCapID: int
  SlideID: str
  SampleID: int
  SlideID1: str
  SampleID1: int
  @property
  def ppscale(self): return floattoint(self.zoomfactor)
  @ppscale.setter
  def ppscale(self, ppscale):
    try:
      pscale = self.pscale
    except AttributeError:
      pscale = None
    try:
      zoomedscale = self.zoomedscale
    except AttributeError:
      zoomedscale = None
    if pscale is zoomedscale is None: assert False
    if pscale is None: self.pscale = zoomedscale * ppscale
    if zoomedscale is None: self.zoomedscale = pscale / ppscale
    assert ppscale == pscale / zoomedscale
  ppscale: int = MetaDataAnnotation(ppscale, writefunction=floattoint, use_default=False)
  a: float
  b: float
  c: float
  d: float
  e: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale")
  f: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale")
  cov_a_a: float
  cov_a_b: float
  cov_a_c: float
  cov_a_d: float
  cov_a_e: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale")
  cov_a_f: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale")
  cov_b_b: float
  cov_b_c: float
  cov_b_d: float
  cov_b_e: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale")
  cov_b_f: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale")
  cov_c_c: float
  cov_c_d: float
  cov_c_e: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale")
  cov_c_f: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale")
  cov_d_d: float
  cov_d_e: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale")
  cov_d_f: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale")
  cov_e_e: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale", power=2)
  cov_e_f: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale", power=2)
  cov_f_f: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="zoomedscale", power=2)

  @classmethod
  def transforminitargs(cls, **kwargs):
    morekwargs = {}
    if len({"pscale", "zoomedscale", "ppscale"} & set(kwargs)) < 2:
      raise TypeError("Have to give at least 2 of pscale, zoomedscale, ppscale")
    if "pscale" not in kwargs:
      morekwargs["pscale"] = kwargs["zoomedscale"] * kwargs["ppscale"]
    elif "zoomedscale" not in kwargs:
      morekwargs["zoomedscale"] = kwargs["pscale"] / kwargs["ppscale"]
    elif "ppscale" not in kwargs:
      morekwargs["ppscale"] = kwargs["pscale"] / kwargs["zoomedscale"]

    if "matrix" in kwargs:
      (morekwargs["a"], morekwargs["b"], morekwargs["e"]), (morekwargs["c"], morekwargs["d"], morekwargs["f"]), lastrow = kwargs.pop("matrix")
      if not np.all(lastrow == [0, 0, 1]):
        raise ValueError(f"Last row of the matrix is {lastrow}, should be [0, 0, 1]")

      letters = "abcdef"
      covariance = units.covariance_matrix([morekwargs[_] for _ in letters])
      for (i, x), (j, y) in itertools.combinations_with_replacement(enumerate(letters), 2):
        morekwargs[f"cov_{x}_{y}"] = covariance[i,j]
      for letter in letters:
        morekwargs[letter] = units.nominal_values(morekwargs[letter])

    return super().transforminitargs(**kwargs, **morekwargs)
