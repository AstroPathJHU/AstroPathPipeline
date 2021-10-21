import collections, contextlib, cv2, matplotlib.pyplot as plt, methodtools, numpy as np, PIL, skimage.registration, skimage.transform

from ...slides.annowarp.annowarpsample import WSISample
from ...slides.align.computeshift import computeshift, crosscorrelation, OptimizeResult, shiftimg
from ...slides.stitchmask.stitchmasksample import AstroPathTissueMaskSample
from ...utilities.misc import floattoint

class ReadWSISample(WSISample, AstroPathTissueMaskSample):
  def __init__(self, *args, uselogfiles=False, **kwargs):
    super().__init__(*args, uselogfiles=uselogfiles, **kwargs)
  @classmethod
  def logmodule(cls): return "readwsi"
  def run(self): assert False

class RegisterWSIs(contextlib.ExitStack):
  def __init__(self, *args, root1, samp1, zoomroot1, root2, samp2, zoomroot2, **kwargs):
    self.samples = (
      ReadWSISample(root=root1, samp=samp1, zoomroot=zoomroot1),
      ReadWSISample(root=root2, samp=samp2, zoomroot=zoomroot2),
    )
    super().__init__(*args, **kwargs)

    self.__nentered = collections.defaultdict(lambda: 0)
    self.__scaledwsisandmasks = {}

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

  def using_scaled_wsis_and_masks(self, *, zoomfactor, smoothsigma, equalize=False):
    return self.__using_scaled_wsis_and_masks(zoomfactor=zoomfactor, smoothsigma=smoothsigma, equalize=equalize)

  def __getscaledwsisandmasks(self, *, zoomfactor, smoothsigma, equalize=False):
    with self.using_wsis() as wsis, self.using_tissuemasks() as masks:
      if zoomfactor > 1:
        wsis = [wsi.resize(np.array(wsi.size)//zoomfactor) for wsi in wsis]
        masks = [skimage.transform.downscale_local_mean(mask, (zoomfactor, zoomfactor)) >= 0.5 for mask in masks]
      masks = [np.asarray(mask) for mask in masks]
      wsis = [np.asarray(wsi) for wsi in wsis]
      if smoothsigma is not None:
        wsis = [skimage.filters.gaussian(wsi, smoothsigma, mode="nearest") for wsi in wsis]
      if equalize:
        wsis = [skimage.exposure.equalize_adapthist(wsi) for wsi in wsis]
      return wsis, masks

  def runalignment(self, *, _debugprint=False, zoomfactor=8, smoothsigma):
    with self.using_scaled_wsis_and_masks(zoomfactor=zoomfactor, smoothsigma=1, equalize=False) as (wsis, masks):
      for _ in wsis:
        print(_.shape, _.dtype)
      for _ in masks:
        print(_.shape, _.dtype)
      wsi1, wsi2 = wsis
      if _debugprint > .5:
        for _ in wsis:
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
        centroid1 += padlow1

      padlow2 = np.max([lowersize-centroid2, [0, 0]], axis=0)
      padhigh2 = np.max([centroid2 + uppersize - wsi2.shape, [0, 0]], axis=0)
      if np.any([padlow2, padhigh2]):
        wsi2 = np.pad(wsi2, ((padlow2[0], padhigh2[0]), (padlow2[1], padhigh2[1])))
        centroid2 += padlow2

      wsis = wsi1, wsi2

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

      wsis = wsi1, wsi2 = wsi1[slice1], wsi2[slice2]

      if _debugprint > .5:
        print("raw wsis")
        for _ in wsis:
          plt.imshow(_)
          plt.show()

      wsis = [skimage.filters.gaussian(wsi, smoothsigma, mode="nearest") for wsi in wsis]

      if _debugprint > .5:
        print("smoothed")
        for _ in wsis:
          plt.imshow(_)
          plt.show()

      wsi1, wsi2 = wsis = [skimage.exposure.equalize_adapthist(wsi) for wsi in wsis]

      if _debugprint > .5:
        print("equalized")
        for _ in wsis:
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

      rotated = wsi1, skimage.transform.rotate(wsi2, rotationresult.angle)

      if _debugprint > .5:
        print("rotated")
        for _ in rotated:
          plt.imshow(_)
          plt.show()

      translationresult = computeshift(rotated[::-1], checkpositivedefinite=False, usemaxmovementcut=False, mindistancetootherpeak=10000, showbigimage=_debugprint>0.5, showsmallimage=_debugprint>0.5)
      return firsttranslationresult, rotationresult, translationresult

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
