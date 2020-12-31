import more_itertools
from ..alignment.field import Field
from ..baseclasses.csvclasses import Polygon, Vertex
from ..baseclasses.rectangle import RectangleReadComponentTiff
from ..baseclasses.sample import ReadRectanglesComponentTiff

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

  def getboundaries(self):
    for field in self.fields:
      mx1 = (field.mx1//self.onepixel)*self.onepixel
      mx2 = (field.mx2//self.onepixel)*self.onepixel
      my1 = (field.my1//self.onepixel)*self.onepixel
      my2 = (field.my2//self.onepixel)*self.onepixel
      Px = mx1, mx2, mx2, mx1
      Py = my1, my1, my2, my2
      fieldvertices = [Vertex(regionid=None, vid=i, x=x, y=y, qpscale=self.pscale) for i, (x, y) in enumerate(more_itertools.zip_equal(Px, Py))]
      fieldpolygon = Polygon(*fieldvertices, pscale=self.pscale)

      with field.using_image() as im:
        contours, hierarchy = cv2.findContours((im==0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchylevels = {-1: -1}
        for i, (contour, (next, previous, child, parent)) in enumerate(more_itertools.zip_equal(contours, hierarchy)):
          assert contour.shape[1:] == (1, 2), contour.shape
          hierarchylevels[i] = hierarchylevel = hierarchylevels[parent]+1
          
