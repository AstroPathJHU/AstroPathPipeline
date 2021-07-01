import contextlib, matplotlib.pyplot as plt, methodtools, numpy as np, skimage.registration, skimage.transform

from ...slides.annowarp.annowarpsample import WSISample
from ...slides.align.computeshift import computeshift
from ...utilities.misc import floattoint

class ReadWSISample(WSISample):
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

  @contextlib.contextmanager
  def using_wsis(self):
    with self.enter_context(contextlib.ExitStack()) as stack:
      wsis = [stack.enter_context(_.using_wsi(1)) for _ in self.samples]
      yield wsis

  @methodtools.lru_cache()
  def scaledwsis(self, zoomfactor, smoothsigma):
    with self.using_wsis() as wsis:
      wsis = [np.asarray(wsi.resize(np.array(wsi.size)//zoomfactor)) for wsi in wsis]
      wsis = [skimage.filters.gaussian(wsi, smoothsigma, mode="nearest") for wsi in wsis]
      return wsis

  def runalignment(self, _debugprint=False, rotationwindowsize=(50, 10), translationwindowsize=10):
    with self.using_wsis() as wsis:
      wsi1, wsi2 = wsis = self.scaledwsis(16, 10)
      if _debugprint:
        for _ in wsis:
          plt.imshow(_)
          plt.show()

      ysize1, xsize1 = shape1 = wsi1.shape
      ygrid1, xgrid1 = np.mgrid[0:ysize1,0:xsize1]
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

      wsis = wsi1, wsi2 = wsi1[slice1], wsi2[slice2]

      if _debugprint:
        for _ in wsis:
          plt.imshow(_)
          plt.show()

      wsispolar = [skimage.transform.warp_polar(wsi) for wsi in wsis]
      if _debugprint:
        for _ in wsispolar:
          plt.imshow(_)
          plt.show()

      rotationresult = computeshift(wsispolar, usemaxmovementcut=False, windowsize=rotationwindowsize, showbigimage=_debugprint, showsmallimage=_debugprint)
      angle = rotationresult.dy
      wsisrotated = [wsis[0], skimage.transform.rotate(wsis[1], angle.n)]
      if _debugprint:
        for _ in wsisrotated:
          plt.imshow(_)
          plt.show()
      translationresult = computeshift(wsisrotated, usemaxmovementcut=False, windowsize=translationwindowsize, showbigimage=_debugprint, showsmallimage=_debugprint)
      return rotationresult, translationresult
