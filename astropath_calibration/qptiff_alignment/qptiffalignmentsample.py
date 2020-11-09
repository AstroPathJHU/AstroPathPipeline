import numpy as np, PIL

from ..zoom.zoom import ZoomSample
from ..baseclasses.qptiff import QPTiff

class QPTiffAlignmentSample(ZoomSample):
  def loadwsi(self, *, layer):
    with self.PILmaximagepixels(), PIL.Image.open(self.wsifilename(layer=layer)) as wsi:
      return np.array(wsi)

  def loadqptiff(self, *, maxwidth, layer):
    with QPTiff(self.qptifffilename) as f:
      for zoomlevel in f.zoomlevels:
        if zoomlevel.imagewidth <= maxwidth:
          break
      return zoomlevel[layer-1].asarray(), zoomlevel.qpscale, f.apscale

  def align(self):
    wsi = self.loadwsi(layer=1)
    qptiff, qpscale, apscale = self.loadqptiff(maxwidth=2000, layer=1)
    pscale = self.pscale
    print(float(qpscale))
    print(float(apscale))
    print(pscale)

  @property
  def logmodule(self):
    return "annowarp"
