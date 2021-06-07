import cv2, methodtools, more_itertools, numpy as np
from ...shared.contours import findcontoursaspolygons
from ...shared.csvclasses import Vertex
from ...shared.polygon import DataClassWithPolygon, SimplePolygon, Polygon, polygonfield
from ...shared.sample import ReadRectanglesDbloadComponentTiff, WorkflowSample
from ...utilities import units
from ...utilities.tableio import writetable
from ..align.alignsample import AlignSample
from ..align.field import FieldReadComponentTiff

class GeomSample(ReadRectanglesDbloadComponentTiff, WorkflowSample):
  """
  The geom step of the pipeline writes out the boundaries of the HPF
  primary regions and the boundaries of the tumor region determined
  by inform to csv files.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, with_seg=True, layer="setlater", **kwargs)
    self.setlayers(layer=self.masklayer)

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
    """
    Find the boundaries of the HPF primary regions.
    """
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
      fieldpolygon = SimplePolygon(vertices=fieldvertices, pscale=self.pscale)
      boundaries.append(Boundary(n=n, k=1, poly=fieldpolygon, pscale=self.pscale, apscale=self.apscale))
    return boundaries

  @methodtools.lru_cache()
  def gettumorboundaries(self):
    """
    Find the boundaries of the tumor region, as determined by
    inform and stored in the mask layer of the segmented component
    tiff.
    """
    self.logger.info("getting tumor boundaries")
    boundaries = []
    for n, field in enumerate(self.rectangles, start=1):
      with field.using_image() as im:
        zeros = im == 0
        if not np.any(zeros): continue
        polygons = findcontoursaspolygons(zeros.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, apscale=self.apscale, shiftby=units.nominal_values(field.pxvec), forgdal=True)
        for k, polygon in enumerate(polygons, start=1):
          boundaries.append(Boundary(n=n, k=k, poly=polygon, pscale=self.pscale, apscale=self.pscale))
    return boundaries

  @property
  def fieldfilename(self): return self.csv("fieldGeometry")
  @property
  def tumorfilename(self): return self.csv("tumorGeometry")

  def writeboundaries(self, *, fieldfilename=None, tumorfilename=None):
    """
    Write the field and tumor boundaries to csv files.
    """
    if fieldfilename is None: fieldfilename = self.fieldfilename
    if tumorfilename is None: tumorfilename = self.tumorfilename
    writetable(fieldfilename, self.getfieldboundaries())
    writetable(tumorfilename, self.gettumorboundaries())

  run = writeboundaries

  def inputfiles(self, **kwargs):
    result = [
      self.csv("constants"),
      self.csv("fields"),
    ]
    if not all(_.exists() for _ in result): return result
    result += [
      *(r.imagefile for r in self.rectangles),
    ]
    result += super().inputfiles(**kwargs)
    return result

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, **otherworkflowkwargs):
    dbload = dbloadroot/SlideID/"dbload"
    return [
      dbload/f"{SlideID}_fieldGeometry.csv",
      dbload/f"{SlideID}_tumorGeometry.csv",
    ]

  @classmethod
  def workflowdependencyclasses(cls):
    return [AlignSample] + super().workflowdependencyclasses()

class Boundary(DataClassWithPolygon):
  """
  Data class for storing a field or tumor boundary.

  n: index of the HPF
  k: index of the boundary within the HPF
  poly: gdal polygon string for the boundary
  """
  n: int
  k: int
  poly: Polygon = polygonfield()

def main(args=None):
  GeomSample.runfromargumentparser(args)

if __name__ == "__main__":
  main()
