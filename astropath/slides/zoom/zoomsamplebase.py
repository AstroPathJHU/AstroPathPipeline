import methodtools, numpy as np
from ...shared.sample import ReadRectanglesDbload
from ...utilities import units
from ...utilities.misc import floattoint, PILmaximagepixels
from ..align.field import Field

class ZoomSampleBase(ReadRectanglesDbload):
  """
  Base class for any sample that does zooming and makes
  a wsi sized image
  """
  rectanglecsv = "fields"
  rectangletype = Field
  def __init__(self, *args, zoomtilesize=16384, **kwargs):
    self.__tilesize = zoomtilesize
    super().__init__(*args, **kwargs)
  multilayer = True
  @property
  def zoomtilesize(self): return self.__tilesize
  @methodtools.lru_cache()
  @property
  def ntiles(self):
    maxxy = np.max([units.nominal_values(field.pxvec)+field.shape+self.margin for field in self.rectangles], axis=0)
    return floattoint(-((-maxxy) // (self.zoomtilesize*self.onepixel)).astype(float))
  def PILmaximagepixels(self):
    return PILmaximagepixels(int(np.product(self.ntiles)) * self.__tilesize**2)

