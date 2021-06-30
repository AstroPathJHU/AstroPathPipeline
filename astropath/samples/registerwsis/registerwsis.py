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
      wsis = tuple(stack.enter_context(_.using_wsi(1)) for _ in self.samples)
      yield wsis

  @methodtools.lru_cache()
  def scaledwsis(self, zoomfactor, smoothsigma):
    with self.using_wsis() as wsis:
      wsis = tuple(np.asarray(wsi.resize(np.array(wsi.size)//zoomfactor)) for wsi in wsis)
      wsis = tuple(skimage.filters.gaussian(wsis, smoothsigma, mode="nearest") for wsi in wsis)
      return wsis

  def runalignment(self):
    with self.using_wsis() as wsis:
      wsi1, wsi2 = wsis = self.scaledwsis(16, 10)
      for _ in wsis:
        plt.imshow(_)
        plt.show()

      ysize1, xsize1 = shape1 = wsi1.shape
      ygrid1, xgrid1 = np.mgrid[0:ysize1,0:xsize1]
      sumwsi1 = np.sum(wsi1)
      centroid1 = floattoint(np.rint(np.array([np.sum(wsi1*ygrid1) / sumwsi1, np.sum(wsi1*xgrid1) / sumwsi1])))

      ysize2, xsize2 = shape2 = wsi2.shape
      ygrid2, xgrid2 = np.mgrid[0:ysize2,0:xsize2]
      sumwsi2 = np.sum(wsi2)
      centroid2 = floattoint(np.rint(np.array([np.sum(wsi2*ygrid2) / sumwsi2, np.sum(wsi2*xgrid2) / sumwsi2])))

      lowersize = np.min([centroid1, centroid2], axis=0)
      uppersize = np.min([shape1-centroid1, shape2-centroid2], axis=0)

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
      print(centroid1, centroid2)
      print(lowersize, uppersize)
      print(shape1, shape2)
      print(slice1, slice2)

      wsis = wsi1, wsi2 = wsi1[slice1], wsi2[slice2]
      print(wsi1.shape, wsi2.shape)

      for _ in wsis:
        plt.imshow(_)
        plt.show()
      wsispolar = tuple(skimage.transform.warp_polar(wsi) for wsi in wsis)
      for _ in wsispolar:
        plt.imshow(_)
        plt.show()
      rotationresult = computeshift(wsispolar, usemaxmovementcut=False, showbigimage=True, showsmallimage=True)
      angle = rotationresult.dy
      wsisrotated = (wsis[0], skimage.transform.rotate(wsis[1], angle.n))
      for _ in wsisrotated:
        plt.imshow(_)
        plt.show()
      translationresult = computeshift(wsisrotated, usemaxmovementcut=False, showbigimage=True, showsmallimage=True)
      return rotationresult, translationresult
