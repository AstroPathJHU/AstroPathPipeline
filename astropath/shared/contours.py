import cv2, more_itertools
from .polygon import SimplePolygon
from ..utilities import units

def findcontoursaspolygons(*args, pscale, annoscale, imagescale=None, shiftby=0, fill=False, forgdal=False, logger, **kwargs):
  """
  Find the contours in a binary image, like cv2.findContours,
  but returns a list of Polygon objects.

  pscale: im3 pixel/micron scale
  annoscale: annotation pixel/micron scale
  imagescale: scale of the image you're converting to polygons
              (default: pscale)
  shiftby: shift all the vertices by this vector (default: [0, 0])
  fill: fill holes in the polygon? (default: False)
        this is useful when the binary array just has 1 along the perimeter
  forgdal: if this is True, the returned polygons will be compatible
           with gdal, meaning they won't have islands nested in holes.
           If there are any islands, those will be returned as separate
           polygons in the list.  Will also check that the polygons are
           "valid": they can't be 1-dimensional or have 1-dimensional tails,
           and they can't have two components joined at a corner.

  positional arguments and other keyword arguments are passed
  directly to cv2.findContours
  """
  if imagescale is None: imagescale = pscale
  contours, (hierarchy,) = cv2.findContours(*args, **kwargs)
  innerpolygons = [[] for c in contours]
  polygons = [None for c in contours]
  toplevelpolygons = []
  oneimagepixel = units.onepixel(imagescale)
  for i, (contour, (next, previous, child, parent)) in reversed(list(enumerate(more_itertools.zip_equal(contours, hierarchy)))):
    assert contour.shape[1:] == (1, 2), contour.shape
    vertices = units.convertpscale(
      [
        [x, y]
        for ((x, y),) in contour*oneimagepixel+shiftby
      ],
      imagescale,
      annoscale,
    )
    polygon = SimplePolygon(vertexarray=vertices, pscale=pscale, annoscale=annoscale)
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
    toplevelpolygons = [p3 for p in toplevelpolygons for p2 in p.polygonsforgdal for p3 in p2.makevalid(logger=logger)]

  return toplevelpolygons
