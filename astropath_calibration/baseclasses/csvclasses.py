import dataclasses, datetime, functools, matplotlib.patches, numbers, numpy as np, re
from ..utilities import units
from ..utilities.misc import dataclass_dc_init, floattoint
from ..utilities.tableio import readtable
from ..utilities.units.dataclasses import DataClassWithApscale, DataClassWithDistances, DataClassWithPscale, DataClassWithPscaleFrozen, distancefield, pscalefield

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
class Vertex(DataClassWithDistances, units.ThingWithPscale, units.ThingWithApscale):
  pixelsormicrons = "pixels"

  regionid: int
  vid: int
  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int, pscalename="apscale")
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int, pscalename="apscale")
  apscale: float = pscalefield()
  readingfromfile: dataclasses.InitVar[bool] = False
  pscale: float = pscalefield(default=None)

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
        for field in dataclasses.fields(type(vertex))
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
    self.__dc_init__(
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

  def __init__(self, *vertices, subtractpolygons=None, pixels=None, microns=None, pscale=None, apscale=None, power=1):
    if power != 1:
      raise ValueError("Polygon should be inited with power=1")
    if bool(vertices) + (pixels is not None) + (microns is not None) != 1:
      raise ValueError("Should provide exactly one of vertices, pixels, or microns")

    if pixels is not None or microns is not None:
      string = pixels if pixels is not None else microns
      kw = "pixels" if pixels is not None else "microns"
      if kw != self.pixelsormicrons:
        raise ValueError(f"Have to provide {self.pixelsormicrons}, not {kw}")

      regex = r"POLYGON \(\(((?:[0-9]* [0-9]*,)*[0-9]* [0-9]*)\)((?:,\(((?:[0-9]* [0-9]*,)*[0-9]* [0-9]*)\))*)\)"
      match = re.match(regex, string)
      if match is None:
        raise ValueError(f"Unexpected format in polygon:\n{string}\nexpected it to match regex:\n{regex}")
      content = match.group(1)
      intvertices = re.findall(r"[0-9]* [0-9]*", content)
      vertices = []
      if intvertices[-1] == intvertices[0]: del intvertices[-1]
      for i, vertex in enumerate(intvertices, start=1):
        x, y = (int(_)*units.onepixel(pscale) for _ in vertex.split())
        vertices.append(Vertex(im3x=x, im3y=y, vid=i, regionid=None, apscale=apscale, pscale=pscale))

      subtractpolygonstrings = match.group(2).replace(")", "").split(",(")
      assert subtractpolygonstrings[0] == ""
      subtractpolygonstrings = subtractpolygonstrings[1:]
      subtractpolygons = [
        Polygon(
          **{kw: f"POLYGON (({subtractpolygonstring}))"},
          pscale=pscale,
          apscale=apscale,
        ) for subtractpolygonstring in subtractpolygonstrings
      ]

    self.__vertices = vertices

    if subtractpolygons is None:
      subtractpolygons = []
    self.__subtractpolygons = subtractpolygons
    for subtractpolygon in subtractpolygons:
      if subtractpolygon.__subtractpolygons:
        raise ValueError("Can't have multiply nested polygons")

    apscale = {apscale, *(v.apscale for v in vertices), *(p.apscale for p in subtractpolygons)}
    apscale.discard(None)
    if len(apscale) > 1: raise ValueError(f"Inconsistent pscales {pscale}")
    self.__apscale, = apscale

    pscale = {pscale, *(v.pscale for v in vertices), *(p.pscale for p in subtractpolygons)}
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
    vertices = list(self.vertices) + [self.vertices[0]]
    return (
      "POLYGON ("
      + ",".join(
        ["(" + ",".join(f"{int(f(v.x, **kwargs))} {int(f(v.y, **kwargs))}" for v in vertices) + ")"]
        + [re.match("POLYGON \((\([^()]*\))\)$", str(subtractpolygon)).group(1) for subtractpolygon in self.__subtractpolygons]
      ) + ")"
    )

  def __eq__(self, other):
    return self.vertices == other.vertices and self.pscale == other.pscale and self.__subtractpolygons == other.__subtractpolygons

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
      class newcls(dataclasses.dataclass(cls, **self.kwargs), DataClassWithPscale, DataClassWithApscale):
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
              apscale=self.apscale,
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
    vertices = [[v.x, v.y] for v in self.vertices]
    if vertices[-1] != vertices[0]: vertices.append(vertices[0])
    for p in self.__subtractpolygons:
      vertices += [[v.x, v.y] for v in p.vertices] + [vertices[0]]
    return matplotlib.patches.Polygon(
      units.convertpscale(
        vertices,
        self.apscale,
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
