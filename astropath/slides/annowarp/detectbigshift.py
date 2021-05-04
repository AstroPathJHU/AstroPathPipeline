import itertools, numpy as np, PIL, skimage.filters
from ...baseclasses.sample import ReadRectanglesDbloadIm3
from ...utilities import units
from ...utilities.misc import floattoint
from ..align.computeshift import computeshift
from .annowarpsample import QPTiffSample

class DetectBigShiftSample(ReadRectanglesDbloadIm3, QPTiffSample, units.ThingWithZoomedscale):
  def __init__(self, *args, shiftthresholdmicrons=100, filetype="flatWarp", **kwargs):
    self.qptifflayer = 1
    self.im3layer = 1
    super().__init__(*args, filetype=filetype, layer=self.im3layer, **kwargs)
    self.__shiftthresholdmicrons = shiftthresholdmicrons
    if len(self.rectangles) != 1:
      raise ValueError("Please specify a rectangle in selectrectangles")
    self.__shift = None

  def logmodule(self): return "detectshift"

  @property
  def zoomfactor(self): return 5
  @property
  def zoomedscale(self): return self.imscale / self.zoomfactor

  def getshift(self):
    if self.__shift is not None: return self.__shift
    r, = self.rectangles
    with r.using_image() as im, self.using_qptiff() as fqptiff:
      im = PIL.Image.fromarray(im)
      zoomlevel = fqptiff.zoomlevels[0]
      qptiff = zoomlevel[self.qptifflayer-1].asarray()
      qptiff = PIL.Image.fromarray(qptiff)

      #scale them so that they're at the same scale and zoomed in
      zoomfactor = 5
      imsize = np.array(im.size, dtype=np.uint)
      qptiffsize = np.array(qptiff.size, dtype=np.uint)

      imsize //= self.ppscale
      qptiffsize = (qptiffsize * self.iqscale).astype(np.uint)

      imsize //= zoomfactor
      qptiffsize //= zoomfactor

      im = im.resize(imsize, resample=PIL.Image.NEAREST)
      qptiff = qptiff.resize(qptiffsize)

      im = np.asarray(im)
      qptiff = np.asarray(qptiff)

      im = im - skimage.filters.gaussian(im, sigma=20)
      im = skimage.filters.gaussian(im, sigma=3)

      px, py = units.convertpscale(r.xvec - self.position, self.pscale, self.zoomedscale)
      w, h = units.convertpscale([r.w, r.h], self.pscale, self.zoomedscale)

      for Dx, Dy in sorted(itertools.product(range(-2, 3), repeat=2), key=lambda x: sum(np.abs(x))):
        ymin = floattoint((py+Dy*h/2) // self.onezoomedpixel)
        ymax = ymin + im.shape[0]
        xmin = floattoint((px+Dx*w/2) // self.onezoomedpixel)
        xmax = xmin + im.shape[1]

        slicedqptiff = qptiff[ymin:ymax, xmin:xmax]
        slicedqptiff = slicedqptiff - skimage.filters.gaussian(slicedqptiff, sigma=20)
        slicedqptiff = skimage.filters.gaussian(slicedqptiff, sigma=3)
        if np.any(im.shape > slicedqptiff.shape): continue
        np.testing.assert_array_equal(im.shape, slicedqptiff.shape)

        try:
          shiftresult = computeshift((slicedqptiff, im), usemaxmovementcut=False)
        except Exception:
          continue
        if shiftresult.exit != 0: continue
        dxvec = np.array([shiftresult.dx.n, shiftresult.dy.n]) * self.onezoomedpixel
        if np.sum(dxvec**2)**.5 > self.__shiftthresholdmicrons * self.onezoomedmicron:
          self.logger.warning(f"Large shift of {dxvec / self.onezoomedmicron} microns between the qptiff and scan")
        break
      else:
        raise ValueError("Couldn't align im3 to qptiff")

      self.__shift = dxvec
      return self.__shift

  run = getshift
