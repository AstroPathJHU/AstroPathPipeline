import dataclassy, itertools, matplotlib.patches, methodtools, numba as nb, numbers, numpy as np, skimage.draw
from numba.core.errors import TypingError
from ..utilities import units
from ..utilities.misc import floattoint
from ..utilities.dataclasses import MetaDataAnnotation
from ..utilities.units.dataclasses import DataClassWithApscale, DataClassWithPscale

class Polygon(units.ThingWithPscale, units.ThingWithApscale):
  """
  This class represents a polygon with holes in it.
  It's more general than gdal because there can also
  be nested holes (i.e. islands inside the holes).
  You can convert it to gdal format as either a single
  polygon if the holes don't have islands or as a list
  of polygons if they do.

  outerpolygon: a SimplePolygon
  subtractpolygons: a list of Polygons to be subtracted from the outer polygon
  """
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
    """
    Returns a polygon that is the same as self but with
    other as a hole in it.
    """
    if isinstance(other, numbers.Number) and other == 0: return self
    return Polygon(outerpolygon=self.outerpolygon, subtractpolygons=self.subtractpolygons+[other])

  @property
  def areas(self):
    """
    Area of the outer ring and negative areas of the inner rings
    """
    return [*self.outerpolygon.areas, *(-a for p in self.subtractpolygons for a in p.areas)]
  @methodtools.lru_cache()
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
    A list of ogr.Geometry object corresponding to the polygon
    and any islands inside its holes.

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
    """
    A single ogr.Geometry object corresponding to the polygon,
    which only works if none of its holes have islands in them.

    imagescale: the scale to use for converting to pixels
                (default: pscale)
    round: round to the nearest pixel?
    """
    try:
      result, = self.gdalpolygons(**kwargs)
    except ValueError:
      raise ValueError("This polygon has multiple levels of nesting, so it can't be expressed as a single gdal polygon")
    return result

  @property
  def __matplotlibpolygonvertices(self):
    vertices = list(self.outerpolygon.vertexarray)
    if not np.all(vertices[0] == vertices[-1]): vertices.append(vertices[0])
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
    return matplotlib.patches.Polygon(
      (units.convertpscale(
        self.__matplotlibpolygonvertices,
        self.apscale,
        imagescale,
      ) + shiftby) / units.onepixel(imagescale),
      **kwargs,
    )

  def numpyarray(self, *, shape, dtype, imagescale=None, shiftby=0):
    """
    A numpy array corresponding to the polygon, with 1 inside the
    polygon and 0 outside.

    shape: shape of the numpy array
    dtype: dtype of the numpy array
    imagescale: the scale to use for converting to pixels (default: pscale)
    shiftby: vector to shift all the vertices by (default: [0, 0])
    """
    if imagescale is None: imagescale = self.pscale
    array = np.zeros(shape, dtype)
    vv = self.outerpolygon.vertexarray
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

  def __str__(self, *, round=True):
    if any(p.subtractpolygons for p in self.subtractpolygons):
      return self.__repr__(round=round)
    else:
      onepixel = self.onepixel
      result = "POLYGON ("
      for i, p in enumerate([self.outerpolygon] + self.subtractpolygons):
        if i != 0: result += ","
        result += "("
        vertices = units.convertpscale(p.vertexarray, self.apscale, self.pscale)
        if round: vertices = floattoint(((vertices+1e-10*onepixel)//onepixel).astype(float))
        vertices = itertools.chain(vertices, [vertices[0]])
        result += ",".join("{} {}".format(*v) for v in vertices)
        result += ")"
      result += ")"
      return result

  def __repr__(self, *, round=True):
    return f"{type(self).__name__}({self.outerpolygon!r}, [{', '.join(_.__repr__(round=round) for _ in self.subtractpolygons)}])"

class SimplePolygon(Polygon):
  """
  Represents a polygon as a list of vertices in a way that works
  with the units functionality.  Also interfaces to gdal and matplotlib.

  vertices: a list of Vertex objects for the corners of the polygon
  OR
  vertexarray: an array of [x, y] of the vertices
  pscale: pscale of the polygon
  apscale: apscale of the polygon
  """

  def __init__(self, *, vertexarray=None, vertices=None, pscale=None, apscale=None, power=1):
    if power != 1:
      raise ValueError("Polygon should be inited with power=1")

    apscale = {apscale}
    pscale = {pscale}

    if vertexarray is not None is vertices:
      vertexarray = np.array(vertexarray)
    elif vertices is not None is vertexarray:
      vertices = list(vertices)
      vertexarray = np.array([v.xvec for v in vertices])
      apscale |= {v.apscale for v in vertices}
      pscale |= {v.pscale for v in vertices}
    else:
      raise TypeError("Have to provide exactly one of vertices or vertexarray")

    apscale.discard(None)
    if len(apscale) > 1: raise ValueError(f"Inconsistent apscales {apscale}")
    elif not apscale: raise ValueError("No apscale provided")
    self.__apscale, = apscale

    pscale.discard(None)
    if len(pscale) > 1: raise ValueError(f"Inconsistent pscales {pscale}")
    elif not pscale: raise ValueError("No pscale provided")
    self.__pscale, = pscale

    self.__vertices = vertices
    self.__vertexarray = vertexarray

    super().__init__(self, [])

    if len(self.__vertexarray) > 1 and np.all(self.__vertexarray[0] == self.__vertexarray[-1]):
      self.__vertexarray = self.__vertexarray[:-1]
      if self.__vertices is not None:
        self.__vertices = self.__vertices[:-1]
    if self.area < 0:
      self.__vertexarray[1:] = self.__vertexarray[1:][::-1]
      if self.__vertices is not None:
        self.__vertices = [self.__vertices[0]] + self.__vertices[:0:-1]

  @property
  def pscale(self):
    return self.__pscale
  @property
  def apscale(self):
    return self.__apscale

  @property
  def vertexarray(self): return self.__vertexarray
  @property
  def vertices(self):
    if self.__vertices is None:
      from .csvclasses import Vertex
      self.__vertices = [
        Vertex(x=x, y=y, vid=i, regionid=None, apscale=self.apscale, pscale=self.pscale)
        for i, (x, y) in enumerate(self.vertexarray)
      ]
    return self.__vertices

  def __eq__(self, other):
    if other is None: return False
    return np.all(self.vertexarray == other.vertexarray)

  def gdallinearring(self, *, imagescale=None, round=False):
    """
    Convert to a gdal linear ring.

    imagescale: the scale to use for converting to pixels (default: pscale)
    round: round to the nearest pixel (default: False)
    """
    if imagescale is None: imagescale = self.pscale
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for v in itertools.chain(self.vertexarray, [self.vertexarray[0]]):
      point = units.convertpscale(v, self.apscale, imagescale)
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
    """
    Area of the polygon
    """
    return units.convertpscale(
      polygonarea(self.vertexarray),
      self.apscale,
      self.pscale,
      power=2,
    )

  @property
  def perimeters(self):
    return [self.perimeter]
  @property
  def perimeter(self):
    """
    Perimeter of the polygon
    """
    return units.convertpscale(
      polygonperimeter(self.vertexarray),
      self.apscale,
      self.pscale,
    )

  def __repr__(self, *, round=True):
    return f"PolygonFromGdal(pixels={str(self.gdalpolygon(round=round))!r}, pscale={self.pscale}, apscale={self.apscale})"

def PolygonFromGdal(*, pixels, pscale, apscale):
  """
  Create a polygon from a GDAL format string or an
  ogr.Geometry object
  """
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
    for x, y in intvertices:
      x *= units.onepixel(pscale)
      y *= units.onepixel(pscale)
      polyvertices.append([x, y])
    polyvertices = units.convertpscale(polyvertices, pscale, apscale)
    vertices.append(polyvertices)

  outerpolygon = SimplePolygon(vertexarray=vertices[0], pscale=pscale, apscale=apscale)
  subtractpolygons = [SimplePolygon(vertexarray=vertices[1], pscale=pscale, apscale=apscale) for v in vertices[1:]]

  return Polygon(outerpolygon, subtractpolygons)

class DataClassWithPolygon(DataClassWithPscale, DataClassWithApscale):
  """
  Dataclass that has at least one field for a polygon

  Usage:
  class HasPolygon(DataClassWithPolygon):
    poly: Polygon = polygonfield()
  ihaveatriangle = HasPolygon(poly="POLYGON((0 0, 1 1, 1 0))", pscale=2, apscale=1)
  """

  @methodtools.lru_cache()
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
    return str(poly)
  metadata = {
    "writefunction": polywritefunction,
    "readfunction": str,
    "ispolygonfield": True,
    **metadata,
  }
  return MetaDataAnnotation(Polygon, **metadata)

class _OgrImport:
  """
  Helper class to import ogr when needed, but allow using the other
  features of this module even if gdal is not installed
  """
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

def __polygonarea(vertexarray):
  size = len(vertexarray)
  sizeminusone = size-1
  doublearea = 0
  for i in range(size):
    x1, y1 = vertexarray[i]
    if i == sizeminusone:
      x2, y2 = vertexarray[0]
    else:
      x2, y2 = vertexarray[i+1]
    doublearea += x1*y2 - x2*y1
  return doublearea / 2
__polygonarea_njit = nb.njit(__polygonarea)

def __polygonarea_safe(vertexarray):
  global polygonarea
  if units.currentmode == "fast":
    polygonarea = __polygonarea_fast
    return polygonarea(vertexarray)
  return __polygonarea(vertexarray)
def __polygonarea_fast(vertexarray):
  try:
    return __polygonarea_njit(vertexarray)
  except TypingError:
    assert units.currentmode == "safe"
    polygonarea = __polygonarea_safe
    return polygonarea(vertexarray)

polygonarea = __polygonarea_safe

def __polygonperimeter(vertexarray):
  size = len(vertexarray)
  sizeminusone = size-1
  perimeter = 0
  for i in range(size):
    v1 = vertexarray[i]
    if i == sizeminusone:
      v2 = vertexarray[0]
    else:
      v2 = vertexarray[i+1]
    perimeter += np.sum((v1 - v2)**2)**.5
  return perimeter
__polygonperimeter_njit = nb.njit(__polygonperimeter)

def __polygonperimeter_safe(vertexarray):
  global polygonperimeter
  if units.currentmode == "fast":
    polygonperimeter = __polygonperimeter_fast
    return polygonperimeter(vertexarray)
  return __polygonperimeter(vertexarray)
def __polygonperimeter_fast(vertexarray):
  try:
    return __polygonperimeter_njit(vertexarray)
  except TypingError:
    assert units.currentmode == "safe"
    polygonperimeter = __polygonperimeter_safe
    return polygonperimeter(vertexarray)

polygonperimeter = __polygonperimeter_safe
