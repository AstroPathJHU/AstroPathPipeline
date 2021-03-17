import cv2, methodtools, more_itertools, numpy as np
from ..alignment.alignmentset import AlignmentSet
from ..alignment.field import FieldReadComponentTiff
from ..baseclasses.csvclasses import Vertex
from ..baseclasses.polygon import DataClassWithPolygon, Polygon, polygonfield
from ..baseclasses.sample import ReadRectanglesDbloadComponentTiff, WorkflowSample
from ..utilities import units
from ..utilities.tableio import writetable
from .contours import findcontoursaspolygons

class GeomSample(ReadRectanglesDbloadComponentTiff, WorkflowSample):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, layer=9, with_seg=True, **kwargs)

  @classmethod
  def logmodule(self): return "geom"

  @property
  def rectanglecsv(self): return "fields"
  rectangletype = FieldReadComponentTiff
  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "with_seg": True,
    }

  @methodtools.lru_cache()
  def getfieldboundaries(self):
    self.logger.info("getting field boundaries")
    boundaries = []
    for field in self.rectangles:
      n = field.n
      mx1 = (field.mx1//self.onepixel)*self.onepixel
      mx2 = (field.mx2//self.onepixel)*self.onepixel
      my1 = (field.my1//self.onepixel)*self.onepixel
      my2 = (field.my2//self.onepixel)*self.onepixel
      Px = mx1, mx2, mx2, mx1
      Py = my1, my1, my2, my2
      fieldvertices = [Vertex(regionid=None, vid=i, im3x=x, im3y=y, apscale=self.apscale, pscale=self.pscale) for i, (x, y) in enumerate(more_itertools.zip_equal(Px, Py))]
      fieldpolygon = Polygon(vertices=[fieldvertices], pscale=self.pscale)
      boundaries.append(Boundary(n=n, k=1, poly=fieldpolygon, pscale=self.pscale, apscale=self.apscale))
    return boundaries

  @methodtools.lru_cache()
  def gettumorboundaries(self):
    self.logger.info("getting tumor boundaries")
    boundaries = []
    for n, field in enumerate(self.rectangles, start=1):
      with field.using_image() as im:
        zeros = im == 0
        if not np.any(zeros): continue
        polygons = findcontoursaspolygons(zeros.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, apscale=self.apscale, shiftby=units.nominal_values(field.pxvec))
        for k, polygon in enumerate(polygons, start=1):
          boundaries.append(Boundary(n=n, k=k, poly=polygon, pscale=self.pscale, apscale=self.pscale))
    return boundaries

  @property
  def fieldfilename(self): return self.csv("fieldGeometry")
  @property
  def tumorfilename(self): return self.csv("tumorGeometry")

  def writeboundaries(self, *, fieldfilename=None, tumorfilename=None):
    if fieldfilename is None: fieldfilename = self.fieldfilename
    if tumorfilename is None: tumorfilename = self.tumorfilename
    writetable(fieldfilename, self.getfieldboundaries())
    writetable(tumorfilename, self.gettumorboundaries())

  @property
  def inputfiles(self):
    return [
      self.csv("constants"),
      self.csv("fields"),
      *(r.imagefile for r in self.rectangles),
    ]
  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, **otherworkflowkwargs):
    dbload = dbloadroot/SlideID/"dbload"
    return [
      dbload/f"{SlideID}_fieldGeometry.csv",
      dbload/f"{SlideID}_tumorGeometry.csv",
    ]

  @classmethod
  def workflowdependencies(cls):
    return [AlignmentSet] + super().workflowdependencies()

class Boundary(DataClassWithPolygon):
  n: int
  k: int
  poly: polygonfield()
