import abc, collections, dataclasses, datetime, methodtools, numpy as np
from ..prepdb.rectangle import Rectangle
from ..utilities import units
from ..utilities.misc import dataclass_dc_init
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

@dataclass_dc_init
class ShiftedRectangle(Rectangle):
  ix: int
  iy: int
  gc: int
  px: float
  py: float
  mx1: float
  mx2: float
  my1: float
  my2: float
  gx: int
  gy: int

  def __init__(self, *args, rectangle=None, **kwargs):
    return self.__dc_init__(
      *args,
      **{
        field.name: getattr(rectangle, field.name)
        for field  in dataclasses.fields(type(rectangle))
      } if rectangle is not None else {},
      **kwargs
    )
