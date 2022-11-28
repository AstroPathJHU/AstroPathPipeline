import dataclassy, itertools, methodtools, more_itertools, numpy as np
from ...shared.csvclasses import Vertex
from ...shared.polygon import DataClassWithPolygon, Polygon, polygonfield, SimplePolygon
from ...shared.rectangle import Rectangle, RectangleCollection, RectangleList, RectangleReadComponentTiffSingleLayer, RectangleReadComponentTiffMultiLayer, RectangleReadIm3MultiLayer, RectangleReadIm3SingleLayer, RectangleReadSegmentedComponentTiffSingleLayer, RectangleReadSegmentedComponentTiffMultiLayer
from ...shared.overlap import Overlap
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

  ix: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int)
  iy: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int)
  gc: int
  px: units.Distance = distancefield(pixelsormicrons="pixels")
  py: units.Distance = distancefield(pixelsormicrons="pixels")
  cov_x_x: units.Distance = distancefield(pixelsormicrons="pixels", power=2)
  cov_x_y: units.Distance = distancefield(pixelsormicrons="pixels", power=2)
  cov_y_y: units.Distance = distancefield(pixelsormicrons="pixels", power=2)
  mx1: units.Distance = distancefield(pixelsormicrons="pixels")
  mx2: units.Distance = distancefield(pixelsormicrons="pixels")
  my1: units.Distance = distancefield(pixelsormicrons="pixels")
  my2: units.Distance = distancefield(pixelsormicrons="pixels")
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

  @methodtools.lru_cache()
  @property
  def boundary(self):
    my1, mx1, my2, mx2 = self.mxbox // self.onepixel * self.onepixel
    Px = mx1, mx2, mx2, mx1
    Py = my1, my1, my2, my2
    vertices = [Vertex(regionid=None, vid=i, im3x=x, im3y=y, pscale=self.pscale, annoscale=self.pscale) for i, (x, y) in enumerate(more_itertools.zip_equal(Px, Py))]
    poly = SimplePolygon(vertices=vertices, pscale=self.pscale)
    return FieldBoundary(n=self.n, k=1, poly=poly, pscale=self.pscale)

class FieldOverlap(Overlap):
  """
  An Overlap with additional information about the stitching.
  It contains the covariance matrix components describing the correlation
  in the positions of the two HPFs described by the overlap.
  """

  cov_x1_x2: units.Distance = distancefield(pixelsormicrons="pixels", power=2)
  cov_x1_y2: units.Distance = distancefield(pixelsormicrons="pixels", power=2)
  cov_y1_x2: units.Distance = distancefield(pixelsormicrons="pixels", power=2)
  cov_y1_y2: units.Distance = distancefield(pixelsormicrons="pixels", power=2)

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

class FieldReadComponentTiffSingleLayer(Field, RectangleReadComponentTiffSingleLayer):
  """
  A Field that can read a single layer of the component tiff
  """

class FieldReadComponentTiffMultiLayer(Field, RectangleReadComponentTiffMultiLayer):
  """
  A Field that can read multiple layers of the component tiff
  """

class FieldReadSegmentedComponentTiffSingleLayer(Field, RectangleReadSegmentedComponentTiffSingleLayer):
  """
  A Field that can read a single layer of the segmented component tiff
  """

class FieldReadSegmentedComponentTiffMultiLayer(Field, RectangleReadSegmentedComponentTiffMultiLayer):
  """
  A Field that can read multiple layers of the segmented component tiff
  """

class FieldReadIm3SingleLayer(Field, RectangleReadIm3SingleLayer):
  """
  A Field that can read a single layer of the im3
  """

class FieldReadIm3MultiLayer(Field, RectangleReadIm3MultiLayer):
  """
  A Field that can read multiple layers of the im3
  """

class FieldCollection(RectangleCollection):
  def showalignedrectanglelayout(self, *, showplot=None, saveas=None, rid=True):
    import matplotlib.patches as patches, matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    xmin = float("inf") * self.onepixel
    xmax = -float("inf") * self.onepixel
    ymin = float("inf") * self.onepixel
    ymax = -float("inf") * self.onepixel
    for r, color in zip(self.rectangles, itertools.cycle(["red", "blue", "green", "yellow", "magenta", "violet", "cyan"])):
      x, y = xy = units.nominal_values(r.pxvec)
      width, height = shape = r.shape
      xmin = min(xmin, x)
      xmax = max(xmax, x+width)
      ymin = min(ymin, y)
      ymax = max(ymax, y+height)
      patch = patches.Rectangle(xy / r.onepixel, *shape / r.onepixel, color=color, alpha=0.5)
      ax.add_patch(patch)
      if rid: ax.text(*(xy+shape/2) / r.onepixel, str(r.n), ha="center", va="center")

    for r in self.rectangles:
      x, y = xy = np.array([r.mx1, r.my1])
      width, height = shape = np.array([r.mx2 - r.mx1, r.my2 - r.my1])
      patch = patches.Rectangle(xy / r.onepixel, *shape / r.onepixel, edgecolor="white", fill=False)
      ax.add_patch(patch)

    margin = .05
    left = float((xmin - (xmax-xmin)*margin) / r.onepixel)
    right = float((xmax + (xmax-xmin)*margin) / r.onepixel)
    top = float((ymin - (ymax-ymin)*margin) / r.onepixel)
    bottom = float((ymax + (ymax-ymin)*margin) / r.onepixel)

    ax.set_xlim(left=left, right=right)
    ax.set_ylim(top=top, bottom=bottom)
    ax.set_aspect('equal', adjustable='box')

    if showplot is None: showplot = saveas is None
    if showplot:
      plt.show()
    if saveas is not None:
      fig.savefig(saveas)
    if not showplot:
      plt.close()

class FieldList(RectangleList, FieldCollection): pass

class FieldBoundary(DataClassWithPolygon):
  """
  Data class for storing a field boundary.

  n: index of the HPF
  k: index of the boundary within the HPF
  poly: gdal polygon string for the boundary
  """
  @classmethod
  def transforminitargs(cls, *args, pscale, **kwargs):
    if "annoscale" not in kwargs: kwargs["annoscale"] = pscale
    return super().transforminitargs(
      *args,
      pscale=pscale,
      **kwargs,
    )
  n: int
  k: int
  poly: Polygon = polygonfield()
