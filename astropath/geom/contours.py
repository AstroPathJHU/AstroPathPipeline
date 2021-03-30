import cv2, more_itertools
from ..baseclasses.polygon import SimplePolygon
from ..utilities import units

def findcontoursaspolygons(*args, pscale, apscale, shiftby=0, fill=False, forgdal=False, **kwargs):
  contours, (hierarchy,) = cv2.findContours(*args, **kwargs)
  innerpolygons = [[] for c in contours]
  polygons = [None for c in contours]
  toplevelpolygons = []
  onepixel = units.onepixel(pscale)
  for i, (contour, (next, previous, child, parent)) in reversed(list(enumerate(more_itertools.zip_equal(contours, hierarchy)))):
    assert contour.shape[1:] == (1, 2), contour.shape
    vertices = units.convertpscale(
      [
        [x, y]
        for ((x, y),) in contour*onepixel+shiftby
      ],
      pscale,
      apscale,
    )
    polygon = SimplePolygon(vertexarray=vertices, pscale=pscale, apscale=apscale)
    for p in innerpolygons[i]:
      polygon -= p
    polygons[i] = polygon
    if parent == -1:
      #prepend because we are iterating in reversed order
      toplevelpolygons.insert(0, polygon)
    else:
      if fill:
        continue
      #inner rings must have >4 points
      if len(vertices) > 4:
        innerpolygons[parent].insert(0, polygon)

  if forgdal:
    toplevelpolygons = [p2 for p in toplevelpolygons for p2 in p.polygonsforgdal]

  return toplevelpolygons
