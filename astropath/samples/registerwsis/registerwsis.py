import collections, contextlib, itertools, matplotlib.pyplot as plt, numpy as np, scipy.ndimage, skimage.registration, skimage.transform, uncertainties as unc

from ...shared.logging import MultiLogger
from ...slides.align.computeshift import computeshift, crosscorrelation, OptimizeResult, shiftimg
from ...slides.align.overlap import AlignmentComparison
from ...slides.annowarp.annowarpsample import WSISample
from ...slides.stitchmask.stitchmasksample import AstroPathTissueMaskSample
from ...utilities import units
from ...utilities.dataclasses import MetaDataAnnotation
from ...utilities.misc import affinetransformation, covariance_matrix, floattoint
from ...utilities.units import ThingWithPscale, ThingWithScale
from ...utilities.units.dataclasses import distancefield, makedataclasswithpscale

class ReadWSISample(WSISample, AstroPathTissueMaskSample):
  def __init__(self, *args, uselogfiles=False, **kwargs):
    super().__init__(*args, uselogfiles=uselogfiles, **kwargs)
  @classmethod
  def logmodule(cls): return "crossregistration"
  def run(self): assert False

class ThingWithZoomedScale(ThingWithScale, scale="zoomedscale"): pass
DataClassWithZoomedScale, DataClassWithZoomedScaleFrozen = makedataclasswithpscale("DataClassWithZoomedScale", "zoomedscale", ThingWithZoomedScale)

class RegisterWSIs(contextlib.ExitStack, ThingWithPscale, ThingWithZoomedScale):
  def __init__(self, *args, root1, samp1, zoomroot1, root2, samp2, zoomroot2, tilepixels=256, zoomfactor=8, mintissuefraction=0.2, uselogfiles=True, **kwargs):
    self.samples = (
      ReadWSISample(root=root1, samp=samp1, zoomroot=zoomroot1, uselogfiles=uselogfiles),
      ReadWSISample(root=root2, samp=samp2, zoomroot=zoomroot2, uselogfiles=uselogfiles),
    )
    self.__zoomfactor = zoomfactor
    super().__init__(*args, **kwargs)

    self.__nentered = collections.defaultdict(lambda: 0)
    self.__scaledwsisandmasks = {}
    self.__tilesize = tilepixels * self.onezoomedpixel
    self.__mintissuefraction = mintissuefraction
    self.__logger = MultiLogger(*(s.logger for s in self.samples), entermessage=f"cross-registering {self.samples[0].SlideID} and {self.samples[1].SlideID}")

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
    with self.using_wsis() as wsis, self.using_tissuemasks() as masks:
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

  def runalignment(self, *, _debugprint=False, smoothsigma):
    with self.using_scaled_wsis_and_masks(smoothsigma=1, equalize=False) as (wsis, masks):
      wsi1, wsi2 = wsis
      mask1, mask2 = masks
      if _debugprint > .5:
        for _ in wsis+masks:
          print(_.shape, _.dtype)
          plt.imshow(_)
          plt.show()

      ysize1, xsize1 = shape1 = wsi1.shape
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

      ysize2, xsize2 = shape2 = wsi2.shape
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
      firsttranslation = affinetransformation(translation=(firsttranslationresult.dx, firsttranslationresult.dy))

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
      r2 = self.getrotation(zoomevenmore, r1.angle-15, r1.angle+15, 2, _debugprint=_debugprint)
      r3 = self.getrotation(zoommore, r2.angle-2, r2.angle+2, 0.02, _debugprint=_debugprint)
      rotationresult = r3
      rotationresult.xcorr.update(r2.xcorr)
      rotationresult.xcorr.update(r1.xcorr)
      rotation = affinetransformation(rotation=rotationresult.angle)

      wsis = wsi1, wsi2 = wsi1, skimage.transform.rotate(wsi2, rotationresult.angle)
      mask1, mask2 = masks = mask1, skimage.transform.rotate(mask2, rotationresult.angle).astype(bool)

      if _debugprint > .5:
        print("rotated")
        for _ in wsis+masks:
          print(_.shape, _.dtype)
          plt.imshow(_)
          plt.show()

      translationresult = computeshift(wsis[::-1], checkpositivedefinite=False, usemaxmovementcut=False, mindistancetootherpeak=10000, showbigimage=_debugprint>0.5, showsmallimage=_debugprint>0.5)
      translation = affinetransformation(translation=(translationresult.dx, translationresult.dy))

      wsis = wsi1, wsi2 = tuple(shiftimg(wsis, -translationresult.dx.n, -translationresult.dy.n))
      masks = mask1, mask2 = tuple(shiftimg(masks, -translationresult.dx.n, -translationresult.dy.n)>0.5)

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
        xvec = x, y = (ixvec-1) * self.tilesize / 4
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
          zoomedscale=self.zoomedscale,
          tilesize=self.tilesize,
          affinetransformation = translation @ rotation @ firsttranslation,
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
#      if write_result:
#        self.writealignments()

      return results

  @staticmethod
  def getrotation(rotationwsis, minangle, maxangle, stepangle, *, _debugprint=-float("inf")):
    wsi1, wsi2 = rotationwsis
    bestxcorr = {}
    for angle in np.arange(minangle, maxangle+stepangle, stepangle):
      rotated = wsi1, skimage.transform.rotate(wsi2, angle)
      if _debugprint > 100:
        for _ in rotated:
          print(angle)
          plt.imshow(_)
          plt.show()
      xcorr = crosscorrelation(rotated)
      bestxcorr[angle] = np.max(xcorr)
    angle = max(bestxcorr, key=bestxcorr.get)
    return OptimizeResult(
      xcorr=bestxcorr,
      angle=angle,
      bestxcorr=bestxcorr[angle],
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
  affinetransformation: np.ndarray = MetaDataAnnotation(None, includeintable=False)

  @classmethod
  def transforminitargs(cls, *args, **kwargs):
    dxvec = kwargs.pop("dxvec", None)
    morekwargs = {}

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
    if self.affinetransformation is None:
      self.affinetransformation = np.identity(3)
    super().__post_init__(*args, **kwargs)

  @property
  def dxvec(self): return np.array([self.dx, self.dy])
  @property
  def unshifted(self): raise NotImplementedError
