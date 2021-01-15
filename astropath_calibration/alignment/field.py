import dataclassy, numpy as np
from ..baseclasses.rectangle import Rectangle
from ..baseclasses.overlap import Overlap
from ..utilities import units
from ..utilities.units.dataclasses import distancefield

class Field(Rectangle):
  ix: distancefield(pixelsormicrons="pixels", dtype=int)
  iy: distancefield(pixelsormicrons="pixels", dtype=int)
  gc: int
  px: distancefield(pixelsormicrons="pixels")
  py: distancefield(pixelsormicrons="pixels")
  cov_x_x: distancefield(pixelsormicrons="pixels", power=2)
  cov_x_y: distancefield(pixelsormicrons="pixels", power=2)
  cov_y_y: distancefield(pixelsormicrons="pixels", power=2)
  mx1: distancefield(pixelsormicrons="pixels")
  mx2: distancefield(pixelsormicrons="pixels")
  my1: distancefield(pixelsormicrons="pixels")
  my2: distancefield(pixelsormicrons="pixels")
  gx: int
  gy: int

  @classmethod
  def transforminitargs(cls, *args, ixvec=None, pxvec=None, gxvec=None, primaryregionx=None, primaryregiony=None, **kwargs):
    veckwargs = {}
    if ixvec is not None:
      veckwargs["ix"], veckwargs["iy"] = ixvec
    if pxvec is not None:
      veckwargs["px"], veckwargs["py"] = units.nominal_values(pxvec)
      (veckwargs["cov_x_x"], veckwargs["cov_x_y"]), (veckwargs["cov_x_y"], veckwargs["cov_y_y"]) = units.covariance_matrix(pxvec)
    if gxvec is not None:
      veckwargs["gx"], veckwargs["gy"] = gxvec
    if primaryregionx is not None:
      veckwargs["mx1"], veckwargs["mx2"] = primaryregionx
    if primaryregiony is not None:
      veckwargs["my1"], veckwargs["my2"] = primaryregiony
    return super().transforminitargs(
      *args,
      **veckwargs,
      **kwargs,
    )

  def __user_init__(self, *args, **kwargs):
    super().__user_init__(*args, **kwargs)

    nominal = [self.px, self.py]
    covariance = [[self.cov_x_x, self.cov_x_y], [self.cov_x_y, self.cov_y_y]]
    self.pxvec = np.array(units.correlated_distances(distances=nominal, covariance=covariance))

  @property
  def _imshowextent(self):
    return [self.px, self.px+self.w, self.py+self.h, self.py]

class FieldOverlap(Overlap):
  cov_x1_x2: distancefield(pixelsormicrons="pixels", power=2)
  cov_x1_y2: distancefield(pixelsormicrons="pixels", power=2)
  cov_y1_x2: distancefield(pixelsormicrons="pixels", power=2)
  cov_y1_y2: distancefield(pixelsormicrons="pixels", power=2)

  @classmethod
  def transforminitargs(cls, *args, overlap=None, **kwargs):
    overlapkwargs = {}
    if overlap is not None:
      overlapkwargs = {
        "pscale": overlap.pscale,
        "nclip": overlap.nclip,
        "rectangles": overlap.rectangles,
        **{
          field: getattr(overlap, field)
          for field in dataclassy.fields(type(overlap))
        }
      }
    return super().transforminitargs(
      *args,
      **overlapkwargs,
      **kwargs,
    )

