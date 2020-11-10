import numpy as np, PIL

from ..baseclasses.qptiff import QPTiff
from ..zoom.zoom import ZoomSample
from ..utilities.misc import floattoint

class QPTiffAlignmentSample(ZoomSample):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__ppscale = None
    wsilayer=1
    qptifflayer=1
    deltax = units.pixels(pixels=1400, pscale=self.pscale)
    deltay = 2100
    tilesize = 100


  @property
  def ppscale(self):
    if self.__ppscale is None: raise AttributeError("Have to load images before trying to get ppscale")
    return self.__ppscale
  @property
  def xposition(self):
    if self.__xposition is None: raise AttributeError("Have to load images before trying to get xposition")
    return self.__xposition
  @property
  def yposition(self):
    if self.__yposition is None: raise AttributeError("Have to load images before trying to get yposition")
    return self.__yposition

  def loadimages(self):
    with self.PILmaximagepixels(), PIL.Image.open(self.wsifilename(layer=self.wsilayer)) as wsi, QPTiff(self.qptifffilename) as fqptiff:
      zoomlevel = fqptiff.zoomlevels[0]
      qptiff = PIL.Image.fromarray(zoomlevel[self.qptifflayer-1].asarray())
      apscale = zoomlevel.qpscale
      pscale = self.pscale
      ipscale = pscale / apscale
      ppscale = self.__ppscale = floattoint(np.round(float(ipscale)))
      iqscale = ipscale / ppscale
      self.__xposition = fqptiff.xposition
      self.__yposition = fqptiff.yposition

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

    mx1 = min(field.mx1 for field in self.rectangles) / self.ppscale
    mx2 = max(field.mx2 for field in self.rectangles) / self.ppscale
    my1 = min(field.my1 for field in self.rectangles) / self.ppscale
    my2 = max(field.my2 for field in self.rectangles) / self.ppscale

    nx1 = max(floattoint(mx1 // self.deltax), 1)
    nx2 = floattoint(mx2 // self.deltax) + 1
    ny1 = max(1, floattoint(my1 // self.deltay), 1)
    ny2 = floattoint(my2 // self.deltay) + 1

    ex = np.arange(nx1, nx2+1) * self.deltax
    ey = np.arange(ny1, ny2+1) * self.deltay

    #tweak the y position by -900 for the microsocope glitches
    #(from Alex's code.  I don't know what this means.)
    qshifty = 0
    if self.yposition == 0: qshifty = 900

    mx2 = min(mx2, wsi.shape[1] - self.tilesize)
    my2 = min(my2, wsi.shape[0] - self.tilesize)

    n1 = floattoint(my1//self.tilesize)-1
    n2 = floattoint(my2//self.tilesize)+1
    m1 = floattoint(mx1//self.tilesize)-1
    m2 = floattoint(mx2//self.tilesize)+1

    for iy in np.arange(n1, n2+1):
      y = self.tilesize * (iy-1)
      for ix in np.arange(m1, m2+1):
          x = self.tilesize * (ix-1)
          if y+1-qshifty <= 0: return
          wsitile = wsi[y:y+self.tilesize, x:x+self.tilesize]
          print(np.min(wsitile), np.max(wsitile))

  @property
  def logmodule(self):
    return "annowarp"
