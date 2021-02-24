import dataclassy, datetime, numbers, numpy as np
from ..utilities import units
from ..utilities.dataclasses import MetaDataAnnotation, MyDataClass
from ..utilities.misc import floattoint
from ..utilities.tableio import readtable
from ..utilities.units.dataclasses import DataClassWithApscale, DataClassWithDistances, DataClassWithPscale, distancefield, pscalefield
from .polygon import DataClassWithPolygon, Polygon, polygonfield

class ROIGlobals(DataClassWithPscale):
  """
  Global info about an ROI of the microscope scan
  """
  pixelsormicrons = "microns"

  x: distancefield(pixelsormicrons=pixelsormicrons)
  y: distancefield(pixelsormicrons=pixelsormicrons)
  Width: distancefield(pixelsormicrons=pixelsormicrons)
  Height: distancefield(pixelsormicrons=pixelsormicrons)
  Unit: str
  Tc: MetaDataAnnotation(datetime.datetime, readfunction=lambda x: datetime.datetime.fromtimestamp(int(x)), writefunction=lambda x: int(datetime.datetime.timestamp(x)))

class ROIPerimeter(DataClassWithPscale):
  """
  Perimeter of an ROI of the microscope scan
  """
  pixelsormicrons = "microns"

  n: int
  x: distancefield(pixelsormicrons=pixelsormicrons)
  y: distancefield(pixelsormicrons=pixelsormicrons)

class Batch(MyDataClass):
  """
  Global info about a sample

  SampleID: numerical ID of the sample
  Sample: the SlideID string
  Scan: the scan number to be used
  Batch: the id of the batch of samples
  """
  SampleID: int
  Sample: str
  Scan: int
  Batch: int

class QPTiffCsv(DataClassWithPscale):
  """
  Info from the qptiff file

  SampleID: numerical ID of the sample
  SlideID: the SlideID string
  ResolutionUnit: units of the qptiff
  XPosition, YPosition: global coordinates of the qptiff
  XResolution, YResolution: resolution of the image used for the jpg thumbnail in pixels/cm
  qpscale: resolution of the image used for the jpg thumbnail in pixels/micron
  apscale: resolution of the first layer of the qptiff
  fname: the filename of the image
  img: currently a dummy string
  """
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
  """
  An entry in the constants.csv spreadsheet.
  This is a little complicated because there are distances with different powers
  and different pscales.
  """
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
  pscale: pscalefield() = None
  apscale: pscalefield() = None
  qpscale: pscalefield() = None

def constantsdict(filename, *, pscale=None, apscale=None, qpscale=None):
  """
  read constants.csv into a dict {constantname: constantvalue}

  pscale, apscale, and qpscale will be read automatically from the csv
  if not provided as kwargs.
  """
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

@dataclassy.dataclass(unsafe_hash=True)
class RectangleFile(DataClassWithPscale):
  """
  Info about a rectangle im3 file (used for sanity checking the
  HPF info in the annotations).
  """
  pixelsormicrons = "microns"

  cx: distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cy: distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  t: datetime.datetime

  @property
  def cxvec(self):
    return np.array([self.cx, self.cy])

class Annotation(DataClassWithPolygon):
  """
  An annotation from a pathologist.

  sampleid: numerical id of the slide
  layer: layer number of the annotation
  name: name of the annotation, e.g. tuor
  color: color of the annotation
  visible: should it be drawn?
  poly: the gdal polygon for the annotation
  """
  sampleid: int
  layer: int
  name: str
  color: str
  visible: MetaDataAnnotation(bool, readfunction=lambda x: bool(int(x)), writefunction=lambda x: int(x))
  poly: polygonfield()

class Vertex(DataClassWithPscale, DataClassWithApscale):
  """
  A vertex of a polygon.

  regionid: numerical id of the region that this vertex is part of
  vid: numerical id of the vertex within the region
  x, y: coordinates of the vertex

  vertices are defined in qptiff coordinates.  However, you can also
  access im3x, im3y, and im3xvec, and you can use those as keyword
  arguments in the Vertex constructor, if you prefer to work in im3
  coordinates.
  """

  pixelsormicrons = "pixels"

  regionid: int
  vid: int
  pscalename = "apscale"
  x: distancefield(pixelsormicrons=pixelsormicrons, dtype=int, pscalename=pscalename)
  y: distancefield(pixelsormicrons=pixelsormicrons, dtype=int, pscalename=pscalename)
  del pscalename
  pscale = None

  @property
  def xvec(self):
    """[x, y] as a numpy array"""
    return np.array([self.x, self.y])

  @classmethod
  def transforminitargs(cls, *args, pscale=None, apscale=None, im3x=None, im3y=None, im3xvec=None, xvec=None, vertex=None, **kwargs):
    xveckwargs = {}
    vertexkwargs = {}
    im3xykwargs = {}
    im3xveckwargs = {}
    if xvec is not None:
      xveckwargs["x"], xveckwargs["y"] = xvec
    if vertex is not None:
      vertexkwargs = {
        field: getattr(vertex, field)
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
    return super().transforminitargs(
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
    """
    [x, y] as a numpy array in im3 coordinates
    """
    if self.pscale is None:
      raise ValueError("Can't get im3 dimensions if you don't provide a pscale")
    return units.convertpscale(self.xvec, self.apscale, self.pscale)
  @property
  def im3x(self):
    """x in im3 coordinates"""
    return self.xvec[0]
  @property
  def im3y(self):
    """y in im3 coordinates"""
    return self.xvec[1]

class Region(DataClassWithPolygon):
  """
  An annotation region

  regionid: numerical id of the region
  sampleid: numerical id of the sample
  layer: layer number to draw the region on
  rid: id of the region within its layer
  isNeg: is this a region or a hole in a region?
  type: Polygon
  nvert: number of vertices of the region
  poly: gdal polygon for the region
  """

  pixelsormicrons = Polygon.pixelsormicrons

  regionid: int
  sampleid: int
  layer: int
  rid: int
  isNeg: MetaDataAnnotation(bool, readfunction=lambda x: bool(int(x)), writefunction=lambda x: int(x))
  type: str
  nvert: int
  poly: polygonfield()
