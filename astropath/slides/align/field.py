import dataclassy, numpy as np
from ...baseclasses.rectangle import Rectangle, RectangleReadComponentTiff, RectangleReadComponentTiffMultiLayer, RectangleReadIm3, RectangleReadIm3MultiLayer
from ...baseclasses.overlap import Overlap
from ...utilities import units
from ...utilities.units.dataclasses import distancefield

class Field(Rectangle):
  """
  A Rectangle with additional information about its stitched position
  ix, iy: (x, y) in integer pixels
  gc: id of the island this field is in
      islands are defined by contiguous HPFs that can be successfully aligned
  px, py: stitched position of the HPF
  cov_x_x, cov_x_y, cov_y_y: covariance matrix of (px, py)
                             note that the absolute error is likely to be large
                             if you want the error on the relative location of two
                             adjacent HPFs, use these values in conjunction with
                             fieldoverlaps.csv
  mx1, mx2, my1, my2: boundaries of the primary region of this field,
                      calculated by an average over the positions of fields in
                      the island in the same row and column
  gx, gy: index of the row and column within the island
  """

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
  """
  An Overlap with additional information about the stitching.
  It contains the covariance matrix components describing the correlation
  in the positions of the two HPFs described by the overlap.
  """

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
  """
  A Field that can read a single layer of the component tiff
  """

class FieldReadComponentTiffMultiLayer(Field, RectangleReadComponentTiffMultiLayer):
  """
  A Field that can read multiple layers of the component tiff
  """

class FieldReadIm3(Field, RectangleReadIm3):
  """
  A Field that can read a single layer of the im3
  """

class FieldReadIm3MultiLayer(Field, RectangleReadIm3MultiLayer):
  """
  A Field that can read multiple layers of the im3
  """
