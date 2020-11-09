import numpy as np, PIL

from ..baseclasses.qptiff import QPTiff
from ..zoom.zoom import ZoomSample
from ..utilities.misc import floattoint

class QPTiffAlignmentSample(ZoomSample):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__ppscale = None

  @property
  def ppscale(self):
    if self.__ppscale is None:
      raise AttributeError("Have to load images before trying to get ppscale")
    return self.__ppscale

  def loadimages(self):
    wsilayer=1
    qptifflayer=1

    with self.PILmaximagepixels(), PIL.Image.open(self.wsifilename(layer=wsilayer)) as wsi, QPTiff(self.qptifffilename) as fqptiff:
      zoomlevel = fqptiff.zoomlevels[0]
      qptiff = PIL.Image.fromarray(zoomlevel[qptifflayer-1].asarray())
      apscale = zoomlevel.qpscale
      pscale = self.pscale
      ipscale = pscale / apscale
      ppscale = self.__ppscale = floattoint(np.round(float(ipscale)))
      iqscale = ipscale / ppscale

      wsisize = np.array(wsi.size, dtype=np.uint)
      qptiffsize = np.array(qptiff.size, dtype=np.uint)
      wsisize //= ppscale
      qptiffsize = (qptiffsize * iqscale).astype(np.uint)
      wsi = wsi.resize(wsisize)
      qptiff = qptiff.resize(qptiffsize)

      newsize = 0, 0, np.min((wsisize[0], qptiffsize[0])), np.min((wsisize[1], qptiffsize[1]))
      wsi = wsi.crop(newsize)
      qptiff = qptiff.crop(newsize)

      return wsi, qptiff

  def align(self):
    wsi, qptiff = self.loadimages()
    deltax = 1400
    deltay = 2100



  @property
  def logmodule(self):
    return "annowarp"
