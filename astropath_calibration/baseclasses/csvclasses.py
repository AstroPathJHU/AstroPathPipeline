import dataclassy, datetime, functools, itertools, matplotlib.patches, more_itertools, numbers, numpy as np, re
from ..utilities import units
from ..utilities.dataclasses import MetaDataAnnotation, MyDataClass
from ..utilities.misc import floattoint
from ..utilities.tableio import readtable
from ..utilities.units.dataclasses import DataClassWithApscale, DataClassWithDistances, DataClassWithPscale, distancefield, pscalefield

class Globals(DataClassWithPscale):
  pixelsormicrons = "microns"

  x: distancefield(pixelsormicrons=pixelsormicrons)
  y: distancefield(pixelsormicrons=pixelsormicrons)
  Width: distancefield(pixelsormicrons=pixelsormicrons)
  Height: distancefield(pixelsormicrons=pixelsormicrons)
  Unit: str
  Tc: MetaDataAnnotation(datetime.datetime, readfunction=lambda x: datetime.datetime.fromtimestamp(int(x)), writefunction=lambda x: int(datetime.datetime.timestamp(x)))

class Perimeter(DataClassWithPscale):
  pixelsormicrons = "microns"

  n: int
  x: distancefield(pixelsormicrons=pixelsormicrons)
  y: distancefield(pixelsormicrons=pixelsormicrons)

class Batch(MyDataClass):
  SampleID: int
  Sample: str
  Scan: int
  Batch: int

class QPTiffCsv(DataClassWithPscale):
  pixelsormicrons = "microns"

  SampleID: int
  SlideID: str
  ResolutionUnit: str
  XPosition: distancefield(pixelsormicrons=pixelsormicrons)
  YPosition: distancefield(pixelsormicrons=pixelsormicrons)
  XResolution: float
  YResolution: float
  qpscale: float
  apscale: float
  fname: str
  img: str

class Constant(DataClassWithDistances):
  def __intorfloat(string):
    if isinstance(string, np.ndarray): string = string[()]
    if isinstance(string, str):
      try: return int(string)
      except ValueError: return float(string)
    elif isinstance(string, numbers.Number):
      try: return floattoint(string)
      except: return string
    else:
      assert False, (type(string), string)

  name: str
  value: distancefield(
    secondfunction=__intorfloat,
    dtype=__intorfloat,
    power=lambda self: 1 if self.unit in ("pixels", "microns") else 0,
    pixelsormicrons=lambda self: self.unit if self.unit in ("pixels", "microns") else "pixels",
    pscalename=lambda self: {
      "locx": "apscale",
      "locy": "apscale",
      "locz": "apscale",
    }.get(self.name, "pscale")
  )
  unit: str
  description: str
  pscale: pscalefield = None
  apscale: pscalefield = None
  qpscale: pscalefield = None

def constantsdict(filename, *, pscale=None, apscale=None, qpscale=None):
  scalekwargs = {"pscale": pscale, "qpscale": qpscale, "apscale": apscale}

  if any(scale is None for scale in scalekwargs.values()):
    tmp = readtable(filename, Constant, extrakwargs={"pscale": 1, "qpscale": 1, "apscale": 1})
    tmpdict = {_.name: _.value for _ in tmp}
    for scalekwarg, scale in scalekwargs.items():
      if scale is None and scalekwarg in tmpdict:
        scalekwargs[scalekwarg] = tmpdict[scalekwarg]

  constants = readtable(filename, Constant, extrakwargs=scalekwargs)
  dct = {constant.name: constant.value for constant in constants}

  #compatibility
  for constant in constants:
    if constant.name == "flayers" and constant.unit == "pixels":
      dct["flayers"] = units.pixels(dct["flayers"], pscale=pscale)

  return dct

@dataclassy.dataclass(frozen=True)
class RectangleFile(DataClassWithPscale):
  pixelsormicrons = "microns"

  cx: distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cy: distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  t: datetime.datetime

  @property
  def cxvec(self):
    return np.array([self.cx, self.cy])

class Annotation(MyDataClass):
  sampleid: int
  layer: int
  name: str
  color: str
  visible: MetaDataAnnotation(bool, readfunction=lambda x: bool(int(x)), writefunction=lambda x: int(x))
  poly: str

class Vertex(DataClassWithPscale, DataClassWithApscale):
  pixelsormicrons = "pixels"

  regionid: int
  vid: int
  x: distancefield(pixelsormicrons=pixelsormicrons, dtype=int, pscalename="apscale")
  y: distancefield(pixelsormicrons=pixelsormicrons, dtype=int, pscalename="apscale")
  pscale = None

  @property
  def xvec(self):
    return np.array([self.x, self.y])

  def __init__(self, *args, pscale=None, apscale=None, im3x=None, im3y=None, im3xvec=None, xvec=None, vertex=None, **kwargs):
    xveckwargs = {}
    vertexkwargs = {}
    im3xykwargs = {}
    im3xveckwargs = {}
    if xvec is not None:
      xveckwargs["x"], xveckwargs["y"] = xvec
    if vertex is not None:
      vertexkwargs = {
        field.name: getattr(vertex, field.name)
        for field in dataclassy.fields(type(vertex))
      }
      if apscale is None: apscale = vertex.apscale
      if apscale != vertex.apscale: raise ValueError(f"Inconsistent apscales {apscale} {vertex.apscale}")
      if pscale is None: pscale = vertex.pscale
      if pscale != vertex.pscale is not None: raise ValueError(f"Inconsistent pscales {pscale} {vertex.pscale}")
      del vertexkwargs["pscale"], vertexkwargs["apscale"]
    if im3x is not None:
      im3xykwargs["x"] = units.convertpscale(im3x, pscale, apscale)
    if im3y is not None:
      im3xykwargs["y"] = units.convertpscale(im3y, pscale, apscale)
    if im3xvec is not None:
      im3xveckwargs["x"], im3xveckwargs["y"] = units.convertpscale(im3xvec, pscale, apscale)
    return super().__init__(
      *args,
      pscale=pscale,
      apscale=apscale,
      **kwargs,
      **xveckwargs,
      **vertexkwargs,
      **im3xykwargs,
      **im3xveckwargs,
    )

  @property
  def im3xvec(self):
    if self.pscale is None:
      raise ValueError("Can't get im3 dimensions if you don't provide a pscale")
    return units.convertpscale(self.xvec, self.apscale, self.pscale)
  @property
  def im3x(self):
    return self.xvec[0]
  @property
  def im3y(self):
    return self.xvec[1]

class Polygon(units.ThingWithPscale, units.ThingWithApscale):
  pixelsormicrons = "pixels"

  def __init__(self, *, vertices=None, pixels=None, microns=None, pscale=None, apscale=None, power=1):
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

      regex = r"POLYGON \((\((?:[0-9]* [0-9]*,)*[0-9]* [0-9]*\)(?:(?:,\((?:(?:[0-9]* [0-9]*,)*[0-9]* [0-9]*)\))*))\)"
      match = re.match(regex, string)
      if match is None:
        raise ValueError(f"Unexpected format in polygon:\n{string}\nexpected it to match regex:\n{regex}")
      content = match.group(1)
      polygons = re.findall(r"\([^\(\)]*\)", content)
      vertices = []
      for polygon in polygons:
        polyvertices = []
        vertices.append(polyvertices)
        intvertices = re.findall(r"[0-9]* [0-9]*", polygon)
        if intvertices[-1] == intvertices[0]: del intvertices[-1]
        for i, vertex in enumerate(intvertices, start=1):
          x, y = (int(_)*units.onepixel(pscale) for _ in vertex.split())
          polyvertices.append(Vertex(im3x=x, im3y=y, vid=i, regionid=None, apscale=apscale, pscale=pscale))

    self.__vertices = [[v for v in vv] for vv in vertices]
    for vv in self.__vertices:
      if len(vv) > 1 and vv[0] == vv[-1]: del vv[-1]

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
    if self.__pscale is None:
      raise AttributeError("Didn't set pscale for this polygon")
    return self.__pscale
  @property
  def apscale(self): return self.__apscale

  @property
  def vertices(self): return self.__vertices
  def __repr__(self):
    return self.tostring(pscale=self.pscale, pixelsormicrons=self.pixelsormicrons)
  def tostring(self, *, pixelsormicrons, **kwargs):
    f = lambda distance, **kwargs: (
      {"pixels": units.pixels, "microns": units.microns}[pixelsormicrons](
        units.convertpscale(distance, self.apscale, self.pscale)
      )
    )
    return (
      "POLYGON ("
      + ",".join(
        "(" + ",".join(f"{int(f(v.x, **kwargs))} {int(f(v.y, **kwargs))}" for v in vv+[vv[0]]) + ")"
        for vv in self.vertices
      ) + ")"
    )

  def __eq__(self, other):
    assert self.pscale == other.pscale
    return self.vertices == other.vertices

  def __pos__(self):
    return self
  def __neg__(self):
    return Polygon(vertices=[[vv[0]]+vv[:0:-1] for vv in self.vertices])
  def __add__(self, other):
    if isinstance(other, numbers.Number) and other == 0: return self
    return Polygon(vertices=self.vertices+other.vertices)
  def __radd__(self, other):
    return self + other
  def __sub__(self, other):
    return self + -other

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

  @staticmethod
  def field(**metadata):
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

  class DataClassWithPolygon(MyDataClass):
    @classmethod
    def polygonfields(cls):
      return [field for field in dataclassy.fields(cls) if cls.metadata(field).get("ispolygonfield", False)]

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      for field in self.polygonfields:
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

    #def _distances_passed_to_init(self):
    #  result = super()._distances_passed_to_init()
    #  if not isinstance(self.poly, Polygon): return [*result, self.poly]
    #  vertices = sum(([v.x, v.y] for vv in self.poly.vertices for v in vv), [])
    #  vertices = [_ for _ in vertices if _]
    #  return [*result, *vertices]

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


class Region(Polygon.DataClassWithPolygon):
  pixelsormicrons = Polygon.pixelsormicrons

  regionid: int
  sampleid: int
  layer: int
  rid: int
  isNeg: MetaDataAnnotation(bool, readfunction=lambda x: bool(int(x)), writefunction=lambda x: int(x))
  type: str
  nvert: int
  poly: Polygon.field()
