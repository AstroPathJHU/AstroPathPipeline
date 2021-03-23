import dataclassy, itertools, matplotlib.patches, more_itertools, numbers, numpy as np, skimage.draw
from ..utilities import units
from ..utilities.dataclasses import MetaDataAnnotation
from ..utilities.units.dataclasses import DataClassWithApscale, DataClassWithPscale

class Polygon(units.ThingWithPscale, units.ThingWithApscale):
  def __init__(self, outerpolygon, subtractpolygons):
    self.__outerpolygon = outerpolygon
    self.__subtractpolygons = subtractpolygons
    assert not outerpolygon.subtractpolygons
    pscales = {_.pscale for _ in [outerpolygon, *subtractpolygons]}
    pscales.discard(None)
    try:
      self.__pscale, = pscales
    except ValueError:
      raise ValueError("Inconsistent pscales: {pscales}")
    apscales = {_.apscale for _ in [outerpolygon, *subtractpolygons]}
    apscales.discard(None)
    try:
      self.__apscale, = apscales
    except ValueError:
      raise ValueError("Inconsistent apscales: {apscales}")

  @property
  def pscale(self):
    return self.__pscale
  @property
  def apscale(self):
    return self.__apscale

  @property
  def outerpolygon(self): return self.__outerpolygon
  @property
  def subtractpolygons(self): return self.__subtractpolygons

  def __sub__(self, other):
    if isinstance(other, numbers.Number) and other == 0: return self
    return Polygon(outerpolygon=self.outerpolygon, subtractpolygons=self.subtractpolygons+[other])

  @property
  def areas(self):
    """
    Area of the outer ring and negative areas of the inner rings
    """
    return [*self.outerpolygon.areas, *(-a for p in self.subtractpolygons for a in p.areas)]
  @property
  def area(self):
    """
    Total area of the polygon
    """
    return np.sum(self.areas)
  @property
  def perimeters(self):
    """
    Perimeters of the outer and inner rings
    """
    return [*self.outerpolygon.perimeters, *(a for p in self.subtractpolygons for a in p.perimeters)]
  @property
  def perimeter(self):
    """
    Total perimeter of the polygon
    """
    return np.sum(self.perimeters)

  @property
  def polygonsforgdal(self):
    """
    Returns a list of polygons, possibly with holes,
    but without islands inside the holes.
    """
    result = []
    outerpoly = self.outerpolygon
    subtractpolys = []
    for poly in self.subtractpolygons:
      subtractpolys.append(poly.outerpolygon)
      for poly2 in poly.subtractpolygons:
        result += poly2.polygonsforgdal
    result.insert(0, Polygon(outerpoly, subtractpolys))
    return result

  def gdalpolygons(self, **kwargs):
    """
    An ogr.Geometry object corresponding to the polygon

    imagescale: the scale to use for converting to pixels
                (default: pscale)
    round: round to the nearest pixel?
    """
    if not any(_.subtractpolygons for _ in self.subtractpolygons):
      outerpoly = ogr.Geometry(ogr.wkbPolygon)
      outerpoly.AddGeometry(self.outerpolygon.gdallinearring(**kwargs))
      for poly in self.subtractpolygons:
        outerpoly.AddGeometry(poly.outerpolygon.gdallinearring(**kwargs))
      return [outerpoly]
    return [p.gdalpolygon(**kwargs) for p in self.polygonsforgdal]

  def gdalpolygon(self, **kwargs):
    try:
      result, = self.gdalpolygons(**kwargs)
    except ValueError:
      raise ValueError("This polygon has multiple levels of nesting, so it can't be expressed as a single gdal polygon")
    return result

  @property
  def __matplotlibpolygonvertices(self):
    vertices = list(self.outerpolygon.vertices)
    if vertices[0] != vertices[-1]: vertices.append(vertices[0])
    for poly in self.subtractpolygons:
      vertices += list(poly.__matplotlibpolygonvertices())
    return vertices

  def matplotlibpolygon(self, *, imagescale=None, shiftby=0, **kwargs):
    """
    An matplotlib.patches.Polygon object corresponding to the polygon

    imagescale: the scale to use for converting to pixels
                (default: pscale)
    shiftby: a 2D vector (distance in imscale) to shift all the vertices by
             (default: 0)
    """
    if imagescale is None: imagescale = self.pscale
    vertices = [[v.x, v.y] for v in self.__matplotlibpolygonvertices]
    return matplotlib.patches.Polygon(
      (units.convertpscale(
        vertices,
        self.apscale,
        imagescale,
      ) + shiftby) / units.onepixel(imagescale),
      **kwargs,
    )

  def numpyarray(self, *, shape, dtype, imagescale=None, shiftby=0):
    if imagescale is None: imagescale = self.pscale
    array = np.zeros(shape, dtype)
    vv = np.array([v.xvec for v in self.outerpolygon.vertices])
    vv = (units.convertpscale(vv, self.apscale, imagescale) + shiftby) // units.onepixel(imagescale)
    coordinates = skimage.draw.polygon(r=vv[:, 1], c=vv[:, 0], shape=shape)
    array[coordinates] = 1
    for p in self.subtractpolygons:
      array = array & ~p.numpyarray(
        shape=shape,
        dtype=dtype,
        imagescale=imagescale,
        shiftby=shiftby,
     )
    return array

  def __eq__(self, other):
    if other is None: return False
    assert self.pscale == other.pscale and self.apscale == other.apscale
    return self.outerpolygon == other.outerpolygon and self.subtractpolygons == other.subtractpolygons

  def __str__(self):
    try:
      return str(self.gdalpolygon(round=True))
    except ValueError:
      return repr(self)

  def __repr__(self):
    return f"{type(self).__name__}({self.outerpolygon!r}, [{', '.join(repr(_) for _ in self.subtractpolygons)}])"

class SimplePolygon(Polygon):
  """
  Represents a polygon as a list of vertices in a way that works
  with the units functionality.  Also interfaces to gdal and matplotlib.

  vertices: a list of Vertex objects for the corners of the polygon
  pixels: a string in GDAL format giving the corners of the polygon in pixels
  """

  pixelsormicrons = "pixels"

  def __init__(self, *, vertices=None, pscale=None, apscale=None, power=1):
    if power != 1:
      raise ValueError("Polygon should be inited with power=1")

    self.__vertices = [v for v in vertices]

    apscale = {apscale, *(v.apscale for v in self.__vertices)}
    apscale.discard(None)
    if len(apscale) > 1: raise ValueError(f"Inconsistent pscales {pscale}")
    self.__apscale, = apscale

    pscale = {pscale, *(v.pscale for v in self.__vertices)}
    pscale.discard(None)
    if len(pscale) > 1: raise ValueError(f"Inconsistent pscales {pscale}")
    self.__pscale, = pscale

    super().__init__(self, [])

    if len(self.__vertices) > 1 and np.all(self.__vertices[0].xvec == self.__vertices[-1].xvec): del self.__vertices[-1]
    if self.area < 0:
      self.__vertices[:] = [self.__vertices[0]] + self.__vertices[:0:-1]

  @property
  def pscale(self):
    return self.__pscale
  @property
  def apscale(self):
    return self.__apscale

  @property
  def vertices(self): return self.__vertices
  @property
  def vxvecs(self): return np.array([v.xvec for v in self.vertices])

  def __eq__(self, other):
    if other is None: return False
    return np.all(self.vxvecs == other.vxvecs)

  def gdallinearring(self, *, imagescale=None, round=False):
    if imagescale is None: imagescale = self.pscale
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for v in itertools.chain(self.vertices, [self.vertices[0]]):
      point = units.convertpscale(v.xvec, self.apscale, imagescale)
      onepixel = units.onepixel(imagescale)
      if round:
        point = (point+1e-10*onepixel) // onepixel
      else:
        point = point / onepixel
      ring.AddPoint_2D(*point.astype(float))
    return ring

  @property
  def areas(self):
    return [self.area]

  @property
  def area(self):
    return units.convertpscale(
      1/2 * sum(
        v1.x*v2.y - v2.x*v1.y
        for v1, v2 in more_itertools.pairwise(itertools.chain(self.vertices, [self.vertices[0]]))
      ),
      self.apscale,
      self.pscale,
      power=2,
    )

  @property
  def perimeters(self):
    return [self.perimeter]
  @property
  def perimeter(self):
    return units.convertpscale(
      sum(
        np.sum((v1.xvec - v2.xvec)**2)**.5
        for v1, v2 in more_itertools.pairwise(itertools.chain(self.vertices, [self.vertices[0]]))
      ),
      self.apscale,
      self.pscale,
    )

  def __repr__(self):
    return f"PolygonFromGdal(pixels={str(self.gdalpolygon())!r}, pscale={self.pscale}, apscale={self.apscale})"

def PolygonFromGdal(*, pixels, pscale, apscale):
  from .csvclasses import Vertex
  if isinstance(pixels, ogr.Geometry):
    gdalpolygon = pixels
  else:
    try:
      gdalpolygon = ogr.CreateGeometryFromWkt(pixels)
    except RuntimeError:
      raise ValueError(f"OGR could not handle the polygon string: {pixels}")

  vertices = []
  for polygon in gdalpolygon:
    polyvertices = []
    intvertices = polygon.GetPoints()
    for j, (x, y) in enumerate(intvertices, start=1):
      x *= units.onepixel(pscale)
      y *= units.onepixel(pscale)
      polyvertices.append(Vertex(im3x=x, im3y=y, vid=j, regionid=None, apscale=apscale, pscale=pscale))
    vertices.append(polyvertices)

  outerpolygon = SimplePolygon(vertices=vertices[0])
  subtractpolygons = [SimplePolygon(vertices=v) for v in vertices[1:]]

  return Polygon(outerpolygon, subtractpolygons)

class DataClassWithPolygon(DataClassWithPscale, DataClassWithApscale):
  """
  Dataclass that has at least one field for a polygon

  Usage:
  class HasPolygon(DataClassWithPolygon):
    poly: polygonfield()
  ihaveatriangle = HasPolygon(poly="POLYGON((0 0, 1 1, 1 0))", pscale=2, apscale=1)
  """

  @classmethod
  def polygonfields(cls):
    return [field for field in dataclassy.fields(cls) if cls.metadata(field).get("ispolygonfield", False)]

  def __post_init__(self, *args, **kwargs):
    super().__post_init__(*args, **kwargs)
    for field in self.polygonfields():
      poly = getattr(self, field)
      if isinstance(poly, Polygon):
        pass
      elif poly is None or isinstance(poly, str) and poly == "poly":
        setattr(self, field, None)
      elif isinstance(poly, (str, ogr.Geometry)):
        setattr(self, field, PolygonFromGdal(
          pixels=poly,
          pscale=self.pscale,
          apscale=self.apscale,
        ))
      else:
        raise TypeError(f"Unknown type {type(poly).__name__} for {field}")

def polygonfield(**metadata):
  """
  A field in a dataclass that is going to get a polygon.
  """
  def polywritefunction(poly):
    if poly is None: return "poly"
    return str(poly.gdalpolygon(round=True))
  metadata = {
    "writefunction": polywritefunction,
    "readfunction": str,
    "ispolygonfield": True,
    **metadata,
  }
  return MetaDataAnnotation(Polygon, **metadata)

class _OgrImport:
  def __getattr__(self, attr):
    global ogr
    try:
      from osgeo import ogr
    except ImportError:
      raise ValueError("Please pip install gdal to use this feature")
    else:
      ogr.UseExceptions()
      return getattr(ogr, attr)

ogr = _OgrImport()
