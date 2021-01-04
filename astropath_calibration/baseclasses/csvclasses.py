import dataclasses, datetime, functools, matplotlib.patches, numbers, numpy as np, re
from ..utilities import units
from ..utilities.misc import dataclass_dc_init, floattoint
from ..utilities.tableio import readtable
from ..utilities.units.dataclasses import DataClassWithDistances, DataClassWithPscale, DataClassWithPscaleFrozen, DataClassWithQpscale, distancefield, pscalefield

@dataclasses.dataclass
class Globals(DataClassWithPscale):
  pixelsormicrons = "microns"

  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  Width: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  Height: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  Unit: str
  Tc: datetime.datetime = dataclasses.field(metadata={"readfunction": lambda x: datetime.datetime.fromtimestamp(int(x)), "writefunction": lambda x: int(datetime.datetime.timestamp(x))})
  readingfromfile: dataclasses.InitVar[bool] = False

@dataclasses.dataclass
class Perimeter(DataClassWithPscale):
  pixelsormicrons = "microns"

  n: int
  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  readingfromfile: dataclasses.InitVar[bool] = False

@dataclasses.dataclass
class Batch:
  SampleID: int
  Sample: str
  Scan: int
  Batch: int

@dataclasses.dataclass
class QPTiffCsv(DataClassWithPscale):
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
  pscale: float = pscalefield()
  readingfromfile: dataclasses.InitVar[bool] = False

@dataclasses.dataclass
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
  value: units.Distance = distancefield(
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
  pscale: float = pscalefield(default=None)
  apscale: float = pscalefield(default=None)
  qpscale: float = pscalefield(default=None)
  readingfromfile: dataclasses.InitVar[bool] = False

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

@dataclasses.dataclass(frozen=True)
class RectangleFile(DataClassWithPscaleFrozen):
  pixelsormicrons = "microns"

  cx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  t: datetime.datetime
  pscale: float = pscalefield()
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

@dataclass_dc_init
class Vertex(DataClassWithQpscale):
  pixelsormicrons = "pixels"

  regionid: int
  vid: int
  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int, pscalename="qpscale")
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int, pscalename="qpscale")
  readingfromfile: dataclasses.InitVar[bool] = False

  @property
  def xvec(self):
    return np.array([self.x, self.y])

  def __init__(self, *args, xvec=None, vertex=None, **kwargs):
    xveckwargs = {}
    vertexkwargs = {}
    if xvec is not None:
      xveckwargs["x"], xveckwargs["y"] = xvec
    if vertex is not None:
      vertexkwargs = {
        field.name: getattr(vertex, field.name)
        for field in dataclasses.fields(type(vertex))
      }
    self.__dc_init__(
      *args,
      **kwargs,
      **xveckwargs,
      **vertexkwargs,
    )

class Polygon(units.ThingWithPscale, units.ThingWithQpscale):
  pixelsormicrons = "pixels"

  def __init__(self, *vertices, subtractpolygons=None, pixels=None, microns=None, pscale=None, qpscale=None, power=1):
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
        x = units.convertpscale(units.Distance(pscale=pscale, **{kw: floattoint(x)}), pscale, qpscale)
        y = units.convertpscale(units.Distance(pscale=pscale, **{kw: floattoint(y)}), pscale, qpscale)
        vertices.append(Vertex(x=x, y=y, vid=i, regionid=None, qpscale=qpscale))

    self.__vertices = vertices
    self.__pscale = pscale

    qpscale = {qpscale, *(v.qpscale for v in vertices)}
    qpscale.discard(None)
    if len(qpscale) > 1: raise ValueError(f"Inconsistent pscales {pscale}")
    self.__qpscale, = qpscale

    if subtractpolygons is None:
      subtractpolygons = []
    self.__subtractpolygons = subtractpolygons
    for subtractpolygon in subtractpolygons:
      if subtractpolygon.__subtractpolygons:
        raise ValueError("Can't have multiply nested polygons")

  @property
  def pscale(self):
    if self.__pscale is None:
      raise AttributeError("Didn't set pscale for this polygon")
    return self.__pscale
  @property
  def _pscale(self): return self.__qpscale
  @property
  def qpscale(self): return self.__qpscale

  @property
  def vertices(self): return self.__vertices
  def __repr__(self):
    return self.tostring(pscale=self.pscale, pixelsormicrons=self.pixelsormicrons)
  def tostring(self, *, pixelsormicrons, **kwargs):
    f = lambda distance, **kwargs: (
      {"pixels": units.pixels, "microns": units.microns}[pixelsormicrons](
        units.convertpscale(distance, self.qpscale, self.pscale)
      )
    )
    vertices = list(self.vertices) + [self.vertices[0]]
    return (
      "POLYGON ("
      + ",".join(
        ["(" + ",".join(f"{int(f(v.x, **kwargs))} {int(f(v.y, **kwargs))}" for v in vertices) + ")"]
        + [re.match("POLYGON\((\([^()]*\))\)$", str(subtractpolygon)).group(1) for subtractpolygon in self.__subtractpolygons]
      ) + ")"
    )

  def __eq__(self, other):
    return self.vertices == other.vertices

  @staticmethod
  def field(*args, metadata={}, **kwargs):
    def polywritefunction(poly):
      if poly is None: return "poly"
      return str(poly)
    metadata = {
      "writefunction": polywritefunction,
      "readfunction": str,
      **metadata,
    }
    return dataclasses.field(*args, metadata=metadata, **kwargs)

  class dataclasswithpolygon:
    def __new__(thiscls, decoratedcls=None, **kwargs):
      if decoratedcls is None: return super().__new__(thiscls)
      if kwargs: raise TypeError("Can't call this with both decoratedcls and kwargs")
      return thiscls()(decoratedcls)

    def __init__(self, **kwargs):
      self.kwargs = kwargs

    def __call__(self, cls):
      @dataclasses.dataclass(**self.kwargs)
      class newcls(dataclasses.dataclass(cls, **self.kwargs), DataClassWithPscale, DataClassWithQpscale):
        @property
        def poly(self):
          return self.__poly
        @poly.setter
        def poly(self, poly):
          if isinstance(poly, Polygon):
            self.__poly = poly
          elif poly is None or poly == "poly":
            self.__poly = None
          elif isinstance(poly, str):
            self.__poly = Polygon(
              **{Polygon.pixelsormicrons: poly},
              pscale=self.pscale,
              qpscale=self.qpscale,
            )
          else:
            raise TypeError(f"Unknown type {type(poly).__name__} for poly")

        def _distances_passed_to_init(self):
          if not isinstance(self.poly, Polygon): return self.poly
          result = sum(([v.x, v.y] for v in self.poly.vertices), [])
          result = [_ for _ in result if _]
          return result

      for thing in functools.WRAPPER_ASSIGNMENTS:
        setattr(newcls, thing, getattr(cls, thing))

      return newcls

  def matplotlibpolygon(self, *, imagescale=None, **kwargs):
    if imagescale is None: imagescale = self.pscale
    return matplotlib.patches.Polygon(
      units.convertpscale(
        [[v.x, v.y] for v in self.vertices],
        self.qpscale,
        imagescale,
      ) / units.onepixel(imagescale),
      **kwargs,
    )


@Polygon.dataclasswithpolygon
class Region:
  pixelsormicrons = Polygon.pixelsormicrons

  regionid: int
  sampleid: int
  layer: int
  rid: int
  isNeg: bool = dataclasses.field(metadata={"readfunction": lambda x: bool(int(x)), "writefunction": lambda x: int(x)})
  type: str
  nvert: int
  poly: Polygon = Polygon.field()
  readingfromfile: dataclasses.InitVar[bool] = False
