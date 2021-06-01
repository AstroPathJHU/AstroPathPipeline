import cv2, more_itertools
from .polygon import SimplePolygon
from ...utilities import units

def findcontoursaspolygons(*args, pscale, apscale, shiftby=0, fill=False, forgdal=False, **kwargs):
  """
  Find the contours in a binary image, like cv2.findContours,
  but returns a list of Polygon objects.

  pscale: im3 pixel/micron scale
  apscale: qptiff pixel/micron scale
  shiftby: shift all hte vertices by this vector (default: [0, 0])
  fill: fill holes in the polygon? (default: False)
        this is useful when the binary array just has 1 along the perimeter
  forgdal: if this is True, the returned polygons will be compatible
           with gdal, meaning they won't have islands nested in holes.
           If there are any islands, those will be returned as separate
           polygons in the list.

  positional arguments and other keyword arguments are passed
  directly to cv2.findContours
  """
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
