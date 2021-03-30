import dataclassy, numpy as np
from ..baseclasses.rectangle import Rectangle, RectangleReadComponentTiff, RectangleReadComponentTiffMultiLayer, RectangleReadIm3, RectangleReadIm3MultiLayer
from ..baseclasses.overlap import Overlap
from ..utilities import units
from ..utilities.units.dataclasses import distancefield

class Field(Rectangle):
  __pixelsormicrons = "pixels"
  ix: distancefield(pixelsormicrons=__pixelsormicrons, dtype=int)
  iy: distancefield(pixelsormicrons=__pixelsormicrons, dtype=int)
  gc: int
  px: distancefield(pixelsormicrons=__pixelsormicrons)
  py: distancefield(pixelsormicrons=__pixelsormicrons)
  cov_x_x: distancefield(pixelsormicrons=__pixelsormicrons, power=2)
  cov_x_y: distancefield(pixelsormicrons=__pixelsormicrons, power=2)
  cov_y_y: distancefield(pixelsormicrons=__pixelsormicrons, power=2)
  mx1: distancefield(pixelsormicrons=__pixelsormicrons)
  mx2: distancefield(pixelsormicrons=__pixelsormicrons)
  my1: distancefield(pixelsormicrons=__pixelsormicrons)
  my2: distancefield(pixelsormicrons=__pixelsormicrons)
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

  @property
  def _imshowextent(self):
    return [self.px, self.px+self.w, self.py+self.h, self.py]

  @property
  def pxvec(self):
    nominal = [self.px, self.py]
    covariance = [[self.cov_x_x, self.cov_x_y], [self.cov_x_y, self.cov_y_y]]
    return np.array(units.correlated_distances(distances=nominal, covariance=covariance))
  @pxvec.setter
  def pxvec(self, pxvec):
    self.px, self.py = units.nominal_values(pxvec)
    ((self.cov_x_x, self.cov_x_y), (self.cov_x_y, self.cov_y_y)) = units.covariance_matrix(pxvec)

  @property
  def primaryregionx(self):
    return np.array([self.mx1, self.mx2])
  @primaryregionx.setter
  def primaryregionx(self, primaryregionx):
    self.mx1, self.mx2 = primaryregionx

  @property
  def primaryregiony(self):
    return np.array([self.my1, self.my2])
  @primaryregiony.setter
  def primaryregiony(self, primaryregiony):
    self.my1, self.my2 = primaryregiony

  @property
  def mxbox(self):
    return np.array([self.my1, self.mx1, self.my2, self.mx2])

class FieldOverlap(Overlap):
  __pixelsormicrons = "pixels"
  cov_x1_x2: distancefield(pixelsormicrons=__pixelsormicrons, power=2)
  cov_x1_y2: distancefield(pixelsormicrons=__pixelsormicrons, power=2)
  cov_y1_x2: distancefield(pixelsormicrons=__pixelsormicrons, power=2)
  cov_y1_y2: distancefield(pixelsormicrons=__pixelsormicrons, power=2)

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

class FieldReadComponentTiff(Field, RectangleReadComponentTiff):
  pass

class FieldReadComponentTiffMultiLayer(Field, RectangleReadComponentTiffMultiLayer):
  pass

class FieldReadIm3(Field, RectangleReadIm3):
  pass

class FieldReadIm3MultiLayer(Field, RectangleReadIm3MultiLayer):
  pass
