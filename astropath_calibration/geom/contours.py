import cv2, more_itertools
from ..baseclasses.csvclasses import Polygon, Vertex
from ..utilities import units

def findcontoursaspolygons(*args, pscale, apscale, shiftby=0, **kwargs):
  contours, (hierarchy,) = cv2.findContours(*args, **kwargs)
  innerpolygons = [[] for c in contours]
  polygons = [None for c in contours]
  toplevelpolygons = []
  onepixel = units.onepixel(pscale)
  for i, (contour, (next, previous, child, parent)) in reversed(list(enumerate(more_itertools.zip_equal(contours, hierarchy)))):
    assert contour.shape[1:] == (1, 2), contour.shape
    vertices = [
      Vertex(im3x=x, im3y=y, vid=i, regionid=None, apscale=apscale, pscale=pscale)
      for i, ((x, y),) in enumerate(contour*onepixel+shiftby, start=1)
    ]
    polygon = polygons[i] = Polygon(vertices=[vertices]) - sum(innerpolygons[i])
    if parent == -1:
      #prepend because we are iterating in reversed order
      toplevelpolygons.insert(0, polygon)
    else:
      #inner rings must have >4 points
      if len(vertices) > 4:
        innerpolygons[parent].insert(0, polygon)
  return toplevelpolygons
