import dataclassy, itertools, matplotlib.patches, more_itertools, numbers, numpy as np
from osgeo import ogr
from ..utilities import units
from ..utilities.dataclasses import MetaDataAnnotation
from ..utilities.units.dataclasses import DataClassWithApscale, DataClassWithPscale

class Polygon(units.ThingWithPscale, units.ThingWithApscale):
  pixelsormicrons = "pixels"

  def __init__(self, *, vertices=None, pixels=None, microns=None, pscale=None, apscale=None, power=1):
    from .csvclasses import Vertex
    if power != 1:
      raise ValueError("Polygon should be inited with power=1")
    if bool(vertices) + (pixels is not None) + (microns is not None) != 1:
      raise ValueError("Should provide exactly one of vertices, pixels, or microns")

    if pixels is not None or microns is not None:
      string = pixels if pixels is not None else microns
      kw = "pixels" if pixels is not None else "microns"
      if kw != self.pixelsormicrons:
        raise ValueError(f"Have to provide {self.pixelsormicrons}, not {kw}")
      if apscale is None: raise ValueError("Have to provide apscale if you give a string to Polygon")
      if pscale is None: raise ValueError("Have to provide pscale if you give a string to Polygon")

      gdalpolygon = ogr.CreateGeometryFromWkt(string)
      vertices = []
      for polygon in gdalpolygon:
        polyvertices = []
        vertices.append(polyvertices)
        intvertices = polygon.GetPoints()
        for j, (x, y) in enumerate(intvertices, start=1):
          x *= units.onepixel(pscale)
          y *= units.onepixel(pscale)
          polyvertices.append(Vertex(im3x=x, im3y=y, vid=j, regionid=None, apscale=apscale, pscale=pscale))

    self.__vertices = [[v for v in vv] for vv in vertices]
    for vv in self.__vertices:
      if len(vv) > 1 and np.all(vv[0].xvec == vv[-1].xvec): del vv[-1]
    for i, (vv, area) in enumerate(more_itertools.zip_equal(self.__vertices, self.areas)):
      if i == 0 and area < 0 or i != 0 and area > 0:
        vv[:] = [vv[0]] + vv[:0:-1]

    apscale = {apscale, *(v.apscale for vv in self.__vertices for v in vv)}
    apscale.discard(None)
    if len(apscale) > 1: raise ValueError(f"Inconsistent pscales {pscale}")
    self.__apscale, = apscale

    pscale = {pscale, *(v.pscale for vv in self.__vertices for v in vv)}
    pscale.discard(None)
    if len(pscale) > 1: raise ValueError(f"Inconsistent pscales {pscale}")
    self.__pscale, = pscale

  @property
  def pscale(self):
    return self.__pscale
  @property
  def apscale(self): return self.__apscale

  @property
  def vertices(self): return self.__vertices
  def __repr__(self):
    return str(self.gdalpolygon(round=True))

  def __eq__(self, other):
    assert self.pscale == other.pscale and self.apscale == other.apscale
    return self.gdalpolygon().Equals(other.gdalpolygon())

  def __sub__(self, other):
    if isinstance(other, numbers.Number) and other == 0: return self
    if len(other.vertices) > 1:
      raise ValueError("Can only subtract a polygon with no holes in it")
    return Polygon(vertices=self.vertices+other.vertices)

  @property
  def separate(self):
    return [Polygon(vertices=[vv]) for vv in self.vertices]
  @property
  def areas(self):
    return [
      1/2 * sum(v1.x*v2.y - v2.x*v1.y for v1, v2 in more_itertools.pairwise(itertools.chain(vv, [vv[0]])))
      for vv in self.vertices
    ]
  @property
  def totalarea(self):
    return np.sum(self.areas)

  def gdalpolygon(self, *, imagescale=None, round=False):
    if imagescale is None: imagescale = self.pscale
    poly = ogr.Geometry(ogr.wkbPolygon)
    for vv in self.vertices:
      ring = ogr.Geometry(ogr.wkbLinearRing)
      for v in itertools.chain(vv, [vv[0]]):
        point = units.convertpscale(v.xvec, self.apscale, imagescale)
        if round:
          point = point // units.onepixel(imagescale)
        else:
          point = point / units.onepixel(imagescale)
        ring.AddPoint_2D(*point.astype(float))
      poly.AddGeometry(ring)
    return poly

  def matplotlibpolygon(self, *, imagescale=None, **kwargs):
    if imagescale is None: imagescale = self.pscale
    vertices = []
    for vv in self.vertices:
      newvertices = [[v.x, v.y] for v in vv]
      if newvertices[-1] != newvertices[0]: newvertices.append(newvertices[0])
      vertices += newvertices
    return matplotlib.patches.Polygon(
      units.convertpscale(
        vertices,
        self.apscale,
        imagescale,
      ) / units.onepixel(imagescale),
      **kwargs,
    )

class DataClassWithPolygon(DataClassWithPscale, DataClassWithApscale):
  @classmethod
  def polygonfields(cls):
    return [field for field in dataclassy.fields(cls) if cls.metadata(field).get("ispolygonfield", False)]

  def __user_init__(self, *args, **kwargs):
    super().__user_init__(*args, **kwargs)
    for field in self.polygonfields():
      poly = getattr(self, field)
      if isinstance(poly, Polygon):
        pass
      elif poly is None or poly == "poly":
        setattr(self, field, None)
      elif isinstance(poly, str):
        setattr(self, field, Polygon(
          **{Polygon.pixelsormicrons: poly},
          pscale=self.pscale,
          apscale=self.apscale,
        ))
      else:
        raise TypeError(f"Unknown type {type(poly).__name__} for {field}")

def polygonfield(**metadata):
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
