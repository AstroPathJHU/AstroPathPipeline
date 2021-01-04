import cv2, methodtools, more_itertools, numpy as np
from ..alignment.field import Field
from ..baseclasses.csvclasses import Polygon, Vertex
from ..baseclasses.rectangle import RectangleReadComponentTiff
from ..baseclasses.sample import ReadRectanglesComponentTiff
from ..utilities import units
from ..utilities.tableio import writetable

class FieldReadComponentTiff(Field, RectangleReadComponentTiff):
  pass

class GeomSample(ReadRectanglesComponentTiff):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, layer=9, with_seg=True, **kwargs)

  @property
  def logmodule(self): return "geom"

  multilayer = False
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
    boundaries = []
    for field in self.rectangles:
      n = field.n
      mx1 = (field.mx1//self.onepixel)*self.onepixel
      mx2 = (field.mx2//self.onepixel)*self.onepixel
      my1 = (field.my1//self.onepixel)*self.onepixel
      my2 = (field.my2//self.onepixel)*self.onepixel
      Px = mx1, mx2, mx2, mx1
      Py = my1, my1, my2, my2
      fieldvertices = [Vertex(regionid=None, vid=i, x=x, y=y, qpscale=self.qpscale, pscale=self.pscale) for i, (x, y) in enumerate(more_itertools.zip_equal(Px, Py))]
      fieldpolygon = Polygon(*fieldvertices, pscale=self.pscale)
      boundaries.append(Boundary(n=n, k=1, poly=fieldpolygon, pscale=self.pscale, qpscale=self.qpscale))
    return boundaries

  @methodtools.lru_cache()
  def gettumorboundaries(self):
    boundaries = []
    for n, field in enumerate(self.rectangles, start=1):
      with field.using_image() as im:
        zeros = im == 0
        if not np.any(zeros): continue
        contours, (hierarchy,) = cv2.findContours(zeros.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        subtractpolygons = [[] for c in contours]
        polygons = [None for c in contours]
        toplevelpolygons = []
        for i, (contour, (next, previous, child, parent)) in reversed(list(enumerate(more_itertools.zip_equal(contours, hierarchy)))):
          assert contour.shape[1:] == (1, 2), contour.shape
          vertices = [
            Vertex(x=x, y=y, vid=i, regionid=None, qpscale=self.qpscale, pscale=self.pscale)
            for i, ((x, y),) in enumerate(contour*self.onepixel+units.nominal_values(field.pxvec), start=1)
          ]
          polygon = polygons[i] = Polygon(*vertices, pscale=self.pscale, subtractpolygons=subtractpolygons[i])
          if parent == -1:
            #prepend because we are iterating in reversed order
            toplevelpolygons.insert(0, polygon)
          else:
            #inner rings must have >4 points
            if len(vertices) > 4:
              subtractpolygons[parent].insert(0, polygon)
        for k, polygon in enumerate(toplevelpolygons, start=1):
          boundaries.append(Boundary(n=n, k=k, poly=polygon, pscale=self.pscale, qpscale=self.pscale))
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

@Polygon.dataclasswithpolygon
class Boundary:
  n: int
  k: int
  poly: Polygon = Polygon.field()
