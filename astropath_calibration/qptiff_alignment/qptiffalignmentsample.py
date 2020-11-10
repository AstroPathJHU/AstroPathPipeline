import numpy as np, PIL

from ..baseclasses.qptiff import QPTiff
from ..zoom.zoom import ZoomSample
from ..utilities import units
from ..utilities.misc import floattoint

class QPTiffAlignmentSample(ZoomSample):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__ppscale = None
    self.wsilayer = 1
    self.qptifflayer = 1
    self.deltax = 1400
    self.deltay = 2100
    self.tilesize = 100

  def align(self):
    with self.PILmaximagepixels(), PIL.Image.open(self.wsifilename(layer=self.wsilayer)) as wsi, QPTiff(self.qptifffilename) as fqptiff:
      zoomlevel = fqptiff.zoomlevels[0]
      qptiff = PIL.Image.fromarray(zoomlevel[self.qptifflayer-1].asarray())
      apscale = zoomlevel.qpscale
      pscale = self.pscale
      ipscale = pscale / apscale
      ppscale = self.__ppscale = floattoint(np.round(float(ipscale)))
      iqscale = ipscale / ppscale
      xposition = fqptiff.xposition
      yposition = fqptiff.yposition

      wsisize = np.array(wsi.size, dtype=np.uint)
      qptiffsize = np.array(qptiff.size, dtype=np.uint)
      wsisize //= ppscale
      qptiffsize = (qptiffsize * iqscale).astype(np.uint)
      wsi = wsi.resize(wsisize)
      qptiff = qptiff.resize(qptiffsize)

      newsize = 0, 0, np.min((wsisize[0], qptiffsize[0])), np.min((wsisize[1], qptiffsize[1]))
      wsi = wsi.crop(newsize)
      qptiff = qptiff.crop(newsize)

      imscales = {apscale * iqscale, pscale / ppscale}
      imscale, = imscales

      wsi = np.asarray(wsi)
      qptiff = np.asarray(qptiff)

    onepixel = units.Distance(pixels=1, pscale=imscale)

    mx1 = units.convertpscale(min(field.mx1 for field in self.rectangles), pscale, imscale, 1)
    mx2 = units.convertpscale(max(field.mx2 for field in self.rectangles), pscale, imscale, 1)
    my1 = units.convertpscale(min(field.my1 for field in self.rectangles), pscale, imscale, 1)
    my2 = units.convertpscale(max(field.my2 for field in self.rectangles), pscale, imscale, 1)

    deltax = units.Distance(pixels=self.deltax, pscale=imscale)
    deltay = units.Distance(pixels=self.deltay, pscale=imscale)
    tilesize = units.Distance(pixels=self.tilesize, pscale=imscale)

    nx1 = max(floattoint(mx1 // deltax), 1)
    nx2 = floattoint(mx2 // deltax) + 1
    ny1 = max(1, floattoint(my1 // deltay), 1)
    ny2 = floattoint(my2 // deltay) + 1

    ex = np.arange(nx1, nx2+1) * self.deltax
    ey = np.arange(ny1, ny2+1) * self.deltay

    #tweak the y position by -900 for the microsocope glitches
    #(from Alex's code.  I don't know what this means.)
    qshifty = 0
    if yposition == 0: qshifty = 900

    mx2 = min(mx2, units.Distance(pixels=wsi.shape[1], pscale=imscale) - tilesize)
    my2 = min(my2, units.Distance(pixels=wsi.shape[0], pscale=imscale) - tilesize)

    n1 = floattoint(my1//tilesize)-1
    n2 = floattoint(my2//tilesize)+1
    m1 = floattoint(mx1//tilesize)-1
    m2 = floattoint(mx2//tilesize)+1

    for iy in np.arange(n1, n2+1):
      y = tilesize * (iy-1)
      for ix in np.arange(m1, m2+1):
          x = tilesize * (ix-1)
          if y+onepixel-qshifty <= 0: return
          wsitile = wsi[
            units.pixels(y, pscale=imscale):units.pixels(y+tilesize, pscale=imscale),
            units.pixels(x, pscale=imscale):units.pixels(x+tilesize, pscale=imscale)
          ]
          print(np.min(wsitile), np.max(wsitile))

  @property
  def logmodule(self):
    return "annowarp"
