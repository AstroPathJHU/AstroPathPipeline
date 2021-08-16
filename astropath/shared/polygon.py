import dataclassy, itertools, matplotlib.patches, methodtools, numba as nb, numbers, numpy as np, skimage.draw
from numba.core.errors import TypingError
from ..utilities import units
from ..utilities.misc import floattoint
from ..utilities.dataclasses import MetaDataAnnotation
from ..utilities.units.dataclasses import DataClassWithApscale, DataClassWithPscale

class InvalidPolygonError(Exception):
  def __init__(self, polygon, reason=None):
    self.polygon = polygon
    self.reason = reason
    message = "Polygon is not valid"
    if reason is not None: message += " ("+reason+")"
    message += ": "+str(polygon)
    try:
      self.madevalid = polygon.MakeValid()
    except RuntimeError as e:
      self.madevalid = None
      message += f"\n\nMakeValid failed: {e}"
    else:
      message += "\n\nMakeValid result:"
      for _ in self.madevalid:
        message += "\n"+str(_)
    super().__init__(message)

  @property
  def validcomponents(self):
    if self.madevalid is None: return []
    result = []
    for component in self.madevalid:
      if "MULTI" in str(component):
        for subcomponent in component:
          result.append(subcomponent)
      else:
        result.append(component)
    return result

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

  def checkvalidity(self):
    poly = self.gdalpolygon()
    try:
      bad = not poly.IsValid()
    except Exception as e:
      raise InvalidPolygonError(poly, reason=str(e))
    if bad:
      raise InvalidPolygonError(poly)

  def makevalid(self, *, round=False, imagescale=None):
    try:
      self.checkvalidity()
    except InvalidPolygonError as e:
      if e.madevalid is None:
        raise
      polygons = []
      for component in e.validcomponents:
        if "MULTI" in str(component):
          assert False
        if "LINESTRING" in str(component):
          continue
        elif "POLYGON" in str(component):
          polygons.append(component)
        elif "LINEARRING" in str(component):
          poly = ogr.Geometry(ogr.wkbPolygon)
          poly.AddGeometry(component)
          polygons.append(poly)
        else:
          raise ValueError(f"Unknown component from MakeValid: {component}")
      polygons = [PolygonFromGdal(pixels=p, pscale=self.pscale, apscale=self.apscale, regionid=self.regionid) for p in polygons]
      if round:
        polygons = [p.round(imagescale=imagescale) for p in polygons]
        polygons = [p for p in polygons if len(p.outerpolygon.vertices) >= 3]
      polygons = sum((p.makevalid(round=round, imagescale=imagescale) for p in polygons), [])
      polygons.sort(key=lambda x: x.area, reverse=True)
      return polygons
    else:
      return [self]

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
    shiftby: a 2D vector (distance in imagescale) to shift all the vertices by
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

  def numpyarray(self, *, shape=None, dtype=bool, imagescale=None, shiftby=None, showvertices=False):
    """
    A numpy array corresponding to the polygon, with 1 inside the
    polygon and 0 outside.

    shape: shape of the numpy array (default: automatically determined to include the whole polygon plus a little buffer)
    dtype: dtype of the numpy array (default: bool)
    imagescale: the scale to use for converting to pixels (default: pscale)
    shiftby: vector to shift all the vertices by (default: [0, 0], unless shape is None in which case it is determined automatically)
    showvertices: also show the vertices in the output array
    """
    if imagescale is None: imagescale = self.pscale

    vv = self.outerpolygon.vertexarray

    if shape is None:
      if shiftby is not None:
        raise TypeError("If you provide shiftby, you also have to provide the array shape")
      shiftby = units.convertpscale(-np.min(vv, axis=0), self.apscale, imagescale) + units.onepixel(imagescale)
    if shiftby is None:
      shiftby = 0

    vv = (units.convertpscale(vv, self.apscale, imagescale) + shiftby) // units.onepixel(imagescale)
    if shape is None:
      shape = floattoint(np.max(vv, axis=0).astype(float)[::-1]+2)

    coordinates = skimage.draw.polygon(r=vv[:, 1], c=vv[:, 0], shape=shape)

    array = np.zeros(shape, dtype)
    array[coordinates] = 1
    if showvertices:
      if self.subtractpolygons:
        raise ValueError("Can't do showvertices when a polygon has holes")
      for vertex in floattoint(vv.astype(float)):
        array[vertex[1], vertex[0]] += 2

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

  def __str__(self, *, round=True, _rounded=False):
    if any(p.subtractpolygons for p in self.subtractpolygons):
      return self.__repr__(round=round)
    else:
      if round: return self.round().__str__(round=False, _rounded=True)
      onepixel = self.onepixel
      result = "POLYGON ("
      for i, p in enumerate([self.outerpolygon] + self.subtractpolygons):
        if i != 0: result += ","
        result += "("
        vertices = units.convertpscale(p.vertexarray, self.apscale, self.pscale) / onepixel
        if _rounded:
          vertices = floattoint(vertices.astype(float))
        vertices = itertools.chain(vertices, [vertices[0]])
        result += ",".join("{} {}".format(*v) for v in vertices)
        result += ")"
      result += ")"
      return result

  def __repr__(self, *, round=True):
    return f"{type(self).__name__}({self.outerpolygon!r}, [{', '.join(_.__repr__(round=round) for _ in self.subtractpolygons)}])"

  @property
  def regionid(self):
    regionids = {self.outerpolygon.regionid, *(_.regionid for _ in self.subtractpolygons)}
    try:
      result, = regionids
    except ValueError:
      raise ValueError(f"Inconsistent regionids {regionids}")
    return result
  @regionid.setter
  def regionid(self, regionid):
    self.outerpolygon.regionid = regionid
    for _ in self.subtractpolygons:
      _.regionid = regionid

  def round(self, **kwargs):
    return Polygon(outerpolygon=self.outerpolygon.round(**kwargs), subtractpolygons=[p.round(**kwargs) for p in self.subtractpolygons])

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

  def __init__(self, *, vertexarray=None, vertices=None, pscale=None, apscale=None, power=1, regionid=None, requirevalidity=False):
    if power != 1:
      raise ValueError("Polygon should be inited with power=1")

    apscale = {apscale}
    pscale = {pscale}
    regionid = {regionid}

    if vertexarray is not None is vertices:
      vertexarray = np.array(vertexarray)
    elif vertices is not None is vertexarray:
      vertices = list(vertices)
      vertexarray = np.array([v.xvec for v in vertices])
      apscale |= {v.apscale for v in vertices}
      pscale |= {v.pscale for v in vertices}
      regionid |= {v.regionid for v in vertices}
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

    regionid.discard(None)
    if len(regionid) > 1: raise ValueError(f"Inconsistent regionids {regionid}")
    elif not regionid: regionid = {None}
    self.__regionid, = regionid

    self.__vertices = vertices
    self.__vertexarray = np.array(vertexarray)

    super().__init__(self, [])

    if len(self.__vertexarray) > 1 and np.all(self.__vertexarray[0] == self.__vertexarray[-1]):
      self.__vertexarray = self.__vertexarray[:-1]
      if self.__vertices is not None:
        self.__vertices = self.__vertices[:-1]
    if self.area < 0:
      self.__vertexarray[1:] = self.__vertexarray[1:][::-1]
      if self.__vertices is not None:
        self.__vertices = [self.__vertices[0]] + self.__vertices[:0:-1]

    lexsorted = np.lexsort(self.__vertexarray.T)
    self.__vertexarray = np.roll(self.__vertexarray, -lexsorted[0], axis=0)
    if self.__vertices is not None:
      self.__vertices = np.roll(self.__vertices, -lexsorted[0], axis=0)

    if requirevalidity:
      self.checkvalidity()

  @property
  def pscale(self):
    return self.__pscale
  @property
  def apscale(self):
    return self.__apscale
  @property
  def regionid(self):
    return self.__regionid
  @regionid.setter
  def regionid(self, regionid):
    self.__regionid = regionid
    if self.__vertices is not None:
      for v in self.__vertices:
        v.regionid = regionid

  @property
  def vertexarray(self): return self.__vertexarray
  @property
  def vertices(self):
    if self.__vertices is None:
      from .csvclasses import Vertex
      self.__vertices = [
        Vertex(x=x, y=y, vid=i, regionid=self.regionid, apscale=self.apscale, pscale=self.pscale)
        for i, (x, y) in enumerate(self.vertexarray, start=1)
      ]
    return self.__vertices

  def __len__(self): return len(self.__vertexarray)+1 #include the first and last points separately

  def __eq__(self, other):
    if other is None: return False
    return np.all(self.vertexarray == other.vertexarray)

  def round(self, *, imagescale=None):
    if imagescale is None: imagescale = self.pscale
    onepixel = units.convertpscale(units.onepixel(imagescale), imagescale, self.apscale)
    vertexarray = (self.vertexarray+1e-10*onepixel) // onepixel * onepixel
    return SimplePolygon(vertexarray=vertexarray, pscale=self.pscale, apscale=self.apscale, regionid=self.regionid)

  def gdallinearring(self, *, imagescale=None, round=False, _rounded=False):
    """
    Convert to a gdal linear ring.

    imagescale: the scale to use for converting to pixels (default: pscale)
    round: round to the nearest pixel (default: False)
    """
    if round: return self.round(imagescale=imagescale).gdallinearring(imagescale=imagescale, _rounded=True)
    if imagescale is None: imagescale = self.pscale
    ring = ogr.Geometry(ogr.wkbLinearRing)
    onepixel = units.onepixel(imagescale)
    vertexarray = units.convertpscale(self.vertexarray, self.apscale, imagescale) / onepixel
    if _rounded: vertexarray = floattoint(vertexarray.astype(float))
    for point in itertools.chain(vertexarray, [vertexarray[0]]):
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

def PolygonFromGdal(*, pixels=None, microns=None, pscale, apscale, **kwargs):
  """
  Create a polygon from a GDAL format string or an
  ogr.Geometry object
  """
  if (pixels is not None) + (microns is not None) != 1:
    raise TypeError("Have to provide exactly one of pixels and microns")
  pixelsormicrons = pixels if pixels is not None else microns

  if isinstance(pixelsormicrons, ogr.Geometry):
    gdalpolygon = pixelsormicrons
  else:
    try:
      gdalpolygon = ogr.CreateGeometryFromWkt(pixelsormicrons)
    except RuntimeError:
      raise ValueError(f"OGR could not handle the polygon string: {pixels}")

  xymultiplier = (units.onepixel if pixels is not None else units.onemicron)(pscale)

  vertices = []
  for polygon in gdalpolygon:
    polyvertices = []
    intvertices = polygon.GetPoints()
    for x, y in intvertices:
      x *= xymultiplier
      y *= xymultiplier
      polyvertices.append([x, y])
    polyvertices = units.convertpscale(polyvertices, pscale, apscale)
    vertices.append(polyvertices)

  outerpolygon = SimplePolygon(vertexarray=vertices[0], pscale=pscale, apscale=apscale, **kwargs)
  subtractpolygons = [SimplePolygon(vertexarray=v, pscale=pscale, apscale=apscale, **kwargs) for v in vertices[1:]]

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
          requirevalidity=True,
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
