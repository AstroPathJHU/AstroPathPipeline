import dataclasses
from ..prepdb.rectangle import Rectangle
from ..utilities import units
from ..utilities.misc import dataclass_dc_init
from ..utilities.units.dataclasses import distancefield

@dataclass_dc_init
class ShiftedRectangle(Rectangle):
  ix: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int)
  iy: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int)
  gc: int
  px: units.Distance = distancefield(pixelsormicrons="pixels")
  py: units.Distance = distancefield(pixelsormicrons="pixels")
  mx1: units.Distance = distancefield(pixelsormicrons="pixels")
  mx2: units.Distance = distancefield(pixelsormicrons="pixels")
  my1: units.Distance = distancefield(pixelsormicrons="pixels")
  my2: units.Distance = distancefield(pixelsormicrons="pixels")
  gx: int
  gy: int

  def __init__(self, *args, rectangle=None, ixvec=None, pxvec=None, gxvec=None, primaryregionx=None, primaryregiony=None, **kwargs):
    veckwargs = {}
    rectanglekwargs = {}
    if ixvec is not None:
      veckwargs["ix"], veckwargs["iy"] = ixvec
    if pxvec is not None:
      veckwargs["px"], veckwargs["py"] = units.nominal_values(pxvec)
    if gxvec is not None:
      veckwargs["gx"], veckwargs["gy"] = gxvec
    if rectangle is not None:
      rectanglekwargs = {
        field.name: getattr(rectangle, field.name)
        for field  in dataclasses.fields(type(rectangle))
      }
    if primaryregionx is not None:
      veckwargs["mx1"], veckwargs["mx2"] = primaryregionx
    if primaryregiony is not None:
      veckwargs["my1"], veckwargs["my2"] = primaryregiony
    return self.__dc_init__(
      *args,
      **veckwargs,
      **rectanglekwargs,
      **kwargs,
    )
