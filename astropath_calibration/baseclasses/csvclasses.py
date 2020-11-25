import dataclasses, datetime, numpy as np, re
from ..utilities import units
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

@dataclasses.dataclass
class Globals(DataClassWithDistances):
  pixelsormicrons = "microns"

  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  Width: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  Height: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  Unit: str
  Tc: datetime.datetime = dataclasses.field(metadata={"readfunction": lambda x: datetime.datetime.fromtimestamp(int(x)), "writefunction": lambda x: int(datetime.datetime.timestamp(x))})
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False

@dataclasses.dataclass
class Perimeter(DataClassWithDistances):
  pixelsormicrons = "microns"

  n: int
  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False

@dataclasses.dataclass
class Batch:
  SampleID: int
  Sample: str
  Scan: int
  Batch: int

@dataclasses.dataclass
class QPTiffCsv(DataClassWithDistances):
  pixelsormicrons = "microns"

  SampleID: int
  SlideID: str
  ResolutionUnit: str
  XPosition: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  YPosition: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  XResolution: float
  YResolution: float
  qpscale: float
  apscale: float
  fname: str
  img: str
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False

@dataclasses.dataclass
class Constant:
  def __intorfloat(string):
    assert isinstance(string, str)
    try: return int(string)
    except ValueError: return float(string)

  def __writefunction(value, *, unit, **kwargs):
    if unit == "pixels":
      return units.pixels(value, **kwargs)
    elif unit == "microns":
      return units.microns(value, **kwargs)
    else:
      return value

  name: str
  value: float = dataclasses.field(metadata={"readfunction": __intorfloat, "writefunction": __writefunction, "writefunctionkwargs": lambda self: {"pscale": self.pscale, "unit": self.unit}})
  unit: str
  description: str
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False

  def __post_init__(self, pscale=None, readingfromfile=False):
    if self.unit in ("pixels", "microns"):
      usedistances = False
      if units.currentmode == "safe" and self.value:
        usedistances = isinstance(self.value, units.safe.Distance)
        if usedistances and readingfromfile: assert False #shouldn't be able to happen
        if not usedistances and not readingfromfile:
          raise ValueError("Have to init with readingfromfile=True if you're not providing distances")

      if pscale and usedistances and self.value._pscale != pscale:
        raise units.UnitsError(f"Provided inconsistent pscales: {pscale} {self.value._pscale}")
      if pscale is None and self.value:
        if not usedistances:
          raise TypeError("Have to either provide pscale explicitly or give coordinates in units.Distance form")
      object.__setattr__(self, "pscale", pscale)

      if readingfromfile:
        object.__setattr__(self, "value", units.Distance(pscale=pscale, **{self.unit: self.value}))

@dataclasses.dataclass(frozen=True)
class RectangleFile(DataClassWithDistances):
  pixelsormicrons = "microns"

  cx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  t: datetime.datetime
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False

  @property
  def cxvec(self):
    return np.array([self.cx, self.cy])

@dataclasses.dataclass
class Annotation:
  sampleid: int
  layer: int
  name: str
  color: str
  visible: bool = dataclasses.field(metadata={"readfunction": lambda x: bool(int(x)), "writefunction": lambda x: int(x)})
  poly: str

@dataclasses.dataclass
class Vertex(DataClassWithDistances):
  pixelsormicrons = "microns"

  regionid: int
  vid: int
  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False

  @property
  def xvec(self):
    return np.array([self.x, self.y])

class Polygon:
  pixelsormicrons = "pixels"

  def __init__(self, *vertices, pixels=None, microns=None, pscale=None, power=1):
    if power != 1:
      raise ValueError("Polygon should be inited with power=1")
    if bool(vertices) + (pixels is not None) + (microns is not None) != 1:
      raise ValueError("Should provide exactly one of vertices, pixels, or microns")

    if pixels is not None or microns is not None:
      string = pixels if pixels is not None else microns
      kw = "pixels" if pixels is not None else "microns"
      if kw != self.pixelsormicrons:
        raise ValueError(f"Have to provide {self.pixelsormicrons}, not {kw}")

      regex = r"POLYGON \(\(((?:[0-9]* [0-9]*,)*[0-9]* [0-9]*)\)\)"
      match = re.match(regex, string)
      if match is None:
        raise ValueError(f"Unexpected format in polygon:\n{string}\nexpected it to match regex:\n{regex}")
      content = match.group(1)
      intvertices = re.findall(r"[0-9]* [0-9]*", content)
      vertices = []
      if intvertices[-1] == intvertices[0]: del intvertices[-1]
      for i, vertex in enumerate(intvertices, start=1):
        x, y = vertex.split()
        x = units.Distance(pscale=pscale, **{kw: int(x)})
        y = units.Distance(pscale=pscale, **{kw: int(y)})
        vertices.append(Vertex(x=x, y=y, vid=i, regionid=0, pscale=pscale))

    self.__vertices = vertices
    pscale = {v.pscale for v in vertices}
    if len(pscale) > 1: raise ValueError(f"Inconsistent pscales {pscale}")
    self.__pscale = pscale.pop()

  @property
  def pscale(self): return self.__pscale

  @property
  def vertices(self): return self.__vertices
  def __repr__(self):
    return self.tostring(pscale=self.pscale)
  def tostring(self, **kwargs):
    f = {"pixels": units.pixels, "microns": units.microns}[self.pixelsormicrons]
    vertices = list(self.vertices) + [self.vertices[0]]
    return "POLYGON ((" + ",".join(f"{int(f(v.x, **kwargs))} {int(f(v.y, **kwargs))}" for v in vertices) + "))"

  def __eq__(self, other):
    return self.vertices == other.vertices

@dataclasses.dataclass
class Region(DataClassWithDistances):
  pixelsormicrons = Polygon.pixelsormicrons

  regionid: int
  sampleid: int
  layer: int
  rid: int
  isNeg: bool = dataclasses.field(metadata={"readfunction": lambda x: bool(int(x)), "writefunction": lambda x: int(x)})
  type: str
  nvert: int
  poly: Polygon = distancefield(pixelsormicrons=pixelsormicrons, dtype=str, metadata={"writefunction": Polygon.tostring})
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False

  def _distances_passed_to_init(self):
    if not isinstance(self.poly, Polygon): return self.poly
    result = sum(([v.x, v.y] for v in self.poly.vertices), [])
    result = [_ for _ in result if _]
    return result
