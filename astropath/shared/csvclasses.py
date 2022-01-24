import csv, dataclassy, datetime, methodtools, numbers, numpy as np, pathlib
from ..utilities import units
from ..utilities.dataclasses import MetaDataAnnotation, MyDataClass
from ..utilities.miscmath import floattoint
from ..utilities.tableio import boolasintfield, datefield, optionalfield, readtable
from ..utilities.units.dataclasses import DataClassWithAnnoscale, DataClassWithApscale, DataClassWithDistances, DataClassWithPscale, DataClassWithPscaleFrozen, distancefield, pscalefield
from .polygon import DataClassWithPolygon, Polygon, polygonfield

class ROIGlobals(DataClassWithPscale):
  """
  Global info about an ROI of the microscope scan
  """
  x: units.Distance = distancefield(pixelsormicrons="microns")
  y: units.Distance = distancefield(pixelsormicrons="microns")
  Width: units.Distance = distancefield(pixelsormicrons="microns")
  Height: units.Distance = distancefield(pixelsormicrons="microns")
  Unit: str
  Tc: datetime.datetime = MetaDataAnnotation(readfunction=lambda x: datetime.datetime.fromtimestamp(int(x)), writefunction=lambda x: int(datetime.datetime.timestamp(x)))

class ROIPerimeter(DataClassWithPscale):
  """
  Perimeter of an ROI of the microscope scan
  """
  n: int
  x: units.Distance = distancefield(pixelsormicrons="microns")
  y: units.Distance = distancefield(pixelsormicrons="microns")

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
  SampleID: int
  SlideID: str
  ResolutionUnit: str
  XPosition: units.Distance = distancefield(pixelsormicrons="microns")
  YPosition: units.Distance = distancefield(pixelsormicrons="microns")
  XResolution: float
  YResolution: float
  qpscale: float
  apscale: float
  fname: str
  img: str

class Constant(DataClassWithDistances, units.ThingWithPscale, units.ThingWithApscale, units.ThingWithQpscale):
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

  @classmethod
  def transforminitargs(cls, name, value, unit=None, description=None, **kwargs):
    if unit is None:
      unit = {
        "fwidth": "pixels",
        "fheight": "pixels",
        "flayers": "",
        "locx": "microns",
        "locy": "microns",
        "locz": "microns",
        "xposition": "microns",
        "yposition": "microns",
        "qpscale": "pixels/micron",
        "apscale": "pixels/micron",
        "pscale": "pixels/micron",
        "nclip": "pixels",
        "margin": "pixels",
        "resolutionbits": "",
        "gainfactor": "",
        "binningx": "pixels",
        "binningy": "pixels",
      }[name]
    if description is None:
      description = {
        "fwidth": "field width",
        "fheight": "field height",
        "flayers": "field depth",
        "locx": "xlocation",
        "locy": "ylocation",
        "locz": "zlocation",
        "xposition": "slide x offset",
        "yposition": "slide y offset",
        "qpscale": "scale of the QPTIFF image",
        "apscale": "scale of the QPTIFF image used for annotation",
        "pscale": "scale of the HPF images",
        "nclip": "pixels to clip off the edge after warping",
        "margin": "minimum margin between the tissue and the wsi edge",
        "resolutionbits": "number of significant bits in the im3 files",
        "gainfactor": "the gain of the A/D amplifier for the im3 files",
        "binningx": "the number of adjacent pixels coadded",
        "binningy": "the number of adjacent pixels coadded",
      }[name]
    return super().transforminitargs(name=name, value=value, unit=unit, description=description, **kwargs)

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
  pscale: float = pscalefield(None)
  apscale: float = pscalefield(None)
  qpscale: float = pscalefield(None)

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
      dct["flayers"] = units.pixels(dct["flayers"], pscale=scalekwargs["pscale"])

  return dct

class RectangleFile(DataClassWithPscaleFrozen):
  """
  Info about a rectangle im3 file (used for sanity checking the
  HPF info in the annotations).
  """
  cx: units.Distance = distancefield(pixelsormicrons="microns", dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons="microns", dtype=int)
  t: datetime.datetime

  @property
  def cxvec(self):
    return np.array([self.cx, self.cy])

class AnnotationInfo(DataClassWithPscale):
  """
  """
  sampleid: int
  name: str
  isonwsi: bool = boolasintfield(False)
  xposition: units.Distance = distancefield(None, optional=True, pixelsormicrons="pixels", dtype=int, pscalename="pscale")
  yposition: units.Distance = distancefield(None, optional=True, pixelsormicrons="pixels", dtype=int, pscalename="pscale")
  xmlfile: str
  xmlsha: str
  scanfolder: pathlib.Path = MetaDataAnnotation(includeintable=False)

  @property
  def isonqptiff(self): return not self.isonwsi

  @classmethod
  def transforminitargs(cls, *args, position=None, **kwargs):
    positionkwargs = {}
    if position is not None:
      positionkwargs["xposition"], positionkwargs["yposition"] = position
    return super().transforminitargs(*args, **kwargs, **positionkwargs)

  @property
  def position(self):
    return np.array([self.xposition, self.yposition], dtype=units.unitdtype)
  @position.setter
  def position(self, position):
    self.xposition, self.yposition = position

  @property
  def xmlpath(self):
    return self.scanfolder/self.xmlfile

class Annotation(DataClassWithPolygon, DataClassWithApscale):
  """
  An annotation from a pathologist.

  sampleid: numerical id of the slide
  layer: layer number of the annotation
  name: name of the annotation, e.g. tumor
  color: color of the annotation
  visible: should it be drawn?
  poly: the gdal polygon for the annotation
  """

  sampleid: int
  layer: int
  name: str
  color: str
  visible: bool = boolasintfield()
  poly: Polygon = polygonfield()
  annotationinfo: AnnotationInfo = MetaDataAnnotation(None, includeintable=False)

  def __bool__(self):
    return self.name.lower() != "empty"

  @methodtools.lru_cache()
  @classmethod
  def isannotationnamefromxml(cls, name):
    if name == "empty": return False
    from .annotationpolygonxmlreader import AllowedAnnotation
    annotations = {_ for _ in AllowedAnnotation.allowedannotations() if _.name == name}
    try:
      a, = annotations
    except ValueError:
      if len(annotations) > 1:
        assert False, annotations
      raise ValueError(f"Didn't find an annotation with name {name} in master_annotation_list.csv")
    return a.isfromxml

  @classmethod
  def transforminitargs(cls, *args, annotationinfo=None, **kwargs):
    isfromxml = cls.isannotationnamefromxml(kwargs["name"])
    if isfromxml and annotationinfo is None:
      raise TypeError("Need annotationinfo if annotation is from xml")
    if "annoscale" not in kwargs:
      pscale = kwargs["pscale"]
      apscale = kwargs["apscale"]
      if not isfromxml:
        kwargs["annoscale"] = pscale
      else:
        if annotationinfo.isonwsi:
          kwargs["annoscale"] = pscale/2
        else:
          kwargs["annoscale"] = apscale
    return super().transforminitargs(*args, annotationinfo=annotationinfo, **kwargs)

  @property
  def isfromxml(self):
    return self.isannotationnamefromxml(self.name)
  @property
  def isonwsi(self):
    if not self.isfromxml: return True
    return self.annotationinfo.isonwsi
  @property
  def isonqptiff(self):
    if not self.isfromxml: return False
    return self.annotationinfo.isonqptiff
  @property
  def position(self): return self.annotationinfo.position

class DataClassWithAnnotation(DataClassWithPscale, DataClassWithApscale, DataClassWithAnnoscale):
  annotation: Annotation = MetaDataAnnotation(None, includeintable=False)
  @classmethod
  def transforminitargs(cls, *, annoscale=None, pscale=None, apscale=None, annotation=None, annotations=None, **kwargs):
    if annotation is not None and annotations is not None:
      raise TypeError("Provided both annotation and annotations")

    if annotations is not None:
      annotation, = (annotation for annotation in annotations if annotation.layer == kwargs["regionid"] // 1000)

    if annotation is not None:
      if annoscale is None: annoscale = annotation.annoscale
      if annoscale != annotation.annoscale: raise ValueError(f"Inconsistent annoscales {annoscale} {annotation.annoscale}")
      if pscale is None: pscale = annotation.pscale
      if pscale != annotation.pscale is not None: raise ValueError(f"Inconsistent pscales {pscale} {annotation.pscale}")
      if apscale is None: apscale = annotation.apscale
      if apscale != annotation.apscale is not None: raise ValueError(f"Inconsistent apscales {apscale} {annotation.apscale}")

    return super().transforminitargs(
      pscale=pscale,
      annoscale=annoscale,
      apscale=apscale,
      annotation=annotation,
      **kwargs,
    )

class Vertex(DataClassWithAnnotation):
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

  regionid: int
  vid: int
  x: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int, pscalename="annoscale")
  y: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int, pscalename="annoscale")
  pscale = None

  @property
  def xvec(self):
    """[x, y] as a numpy array"""
    return np.array([self.x, self.y])
  @property
  def isfromxml(self):
    return self.annotation.isfromxml
  @property
  def isonwsi(self):
    return self.annotation.isonwsi

  @classmethod
  def transforminitargs(cls, *, pscale=None, annoscale=None, im3x=None, im3y=None, im3xvec=None, xvec=None, vertex=None, **kwargs):
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
      if annoscale is None: annoscale = vertex.annoscale
      if annoscale != vertex.annoscale: raise ValueError(f"Inconsistent annoscales {annoscale} {vertex.annoscale}")
      if pscale is None: pscale = vertex.pscale
      if pscale != vertex.pscale is not None: raise ValueError(f"Inconsistent pscales {pscale} {vertex.pscale}")
      del vertexkwargs["pscale"], vertexkwargs["annoscale"]
    if im3x is not None:
      im3xykwargs["x"] = units.convertpscale(im3x, pscale, annoscale)
    if im3y is not None:
      im3xykwargs["y"] = units.convertpscale(im3y, pscale, annoscale)
    if im3xvec is not None:
      im3xveckwargs["x"], im3xveckwargs["y"] = units.convertpscale(im3xvec, pscale, annoscale)
    return super().transforminitargs(
      pscale=pscale,
      annoscale=annoscale,
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
    return units.convertpscale(self.xvec, self.annoscale, self.pscale)
  @property
  def im3x(self):
    """x in im3 coordinates"""
    return self.xvec[0]
  @property
  def im3y(self):
    """y in im3 coordinates"""
    return self.xvec[1]

class Region(DataClassWithPolygon, DataClassWithAnnotation):
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

  regionid: int
  sampleid: int
  layer: int
  rid: int
  isNeg: bool = MetaDataAnnotation(readfunction=lambda x: bool(int(x)), writefunction=lambda x: int(x))
  type: str
  nvert: int
  poly: Polygon = polygonfield()

class ExposureTime(DataClassWithPscale):
  """
  The exposure time for a layer of an HPF
  n: the rectangle id
  cx, cy: the coordinates of the HPF center, in integer pixels
  layer: the index of the layer, starting from 1
  exp: the exposure time
  """

  n: int
  cx: units.Distance = distancefield(pixelsormicrons="microns", dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons="microns", dtype=int)
  layer: int
  exp: float

class MergeConfig(MyDataClass):
  """
  Each MergeConfig object represents a row of the mergeconfig.csv file,
  which describes the stains in the slide and how they map to layers
  of the component tiff.
  """
  @classmethod
  def transforminitargs(cls, *args, **kwargs):
    #compatibility
    for kw in kwargs.copy():
      stripped = kw.strip()
      if stripped not in kwargs:
        kwargs[stripped] = kwargs.pop(kw)
    return super().transforminitargs(*args, **kwargs)

  Project: int = None
  Cohort: int = None
  BatchID: int
  layer: int = None
  Opal: str
  Target: str
  Compartment: str
  TargetType: str
  CoexpressionStatus: str
  SegmentationStatus: int
  SegmentationHierarchy: int
  NumberofSegmentations: int
  ImageQA: str

class GlobalBatch(MyDataClass):
  Project: int
  Cohort: int
  BatchID: int
  OpalLot: str
  Opal: str
  OpalDilution: str
  Target: str
  Compartment: str
  AbClone: str
  AbLot: str
  AbDilution: str

class PhenotypedCell(MyDataClass):
  CellID: int
  SlideID: str
  fx: int
  fy: int
  CellNum: int
  Phenotype: str
  CellXPos: int
  CellYPos: int
  EntireCellArea: int
  MeanNucleusDAPI: float
  MeanNucleus480: float
  MeanNucleus520: float
  MeanNucleus540: float
  MeanNucleus570: float
  MeanNucleus620: float
  MeanNucleus650: float
  MeanNucleus690: float
  MeanNucleus780: float
  MeanMembraneDAPI: float = optionalfield(readfunction=float)
  MeanMembrane480: float = optionalfield(readfunction=float)
  MeanMembrane520: float = optionalfield(readfunction=float)
  MeanMembrane540: float = optionalfield(readfunction=float)
  MeanMembrane570: float = optionalfield(readfunction=float)
  MeanMembrane620: float = optionalfield(readfunction=float)
  MeanMembrane650: float = optionalfield(readfunction=float)
  MeanMembrane690: float = optionalfield(readfunction=float)
  MeanMembrane780: float = optionalfield(readfunction=float)
  MeanEntireCellDAPI: float
  MeanEntireCell480: float
  MeanEntireCell520: float
  MeanEntireCell540: float
  MeanEntireCell570: float
  MeanEntireCell620: float
  MeanEntireCell650: float
  MeanEntireCell690: float
  MeanEntireCell780: float
  MeanCytoplasmDAPI: float = optionalfield(readfunction=float)
  MeanCytoplasm480: float = optionalfield(readfunction=float)
  MeanCytoplasm520: float = optionalfield(readfunction=float)
  MeanCytoplasm540: float = optionalfield(readfunction=float)
  MeanCytoplasm570: float = optionalfield(readfunction=float)
  MeanCytoplasm620: float = optionalfield(readfunction=float)
  MeanCytoplasm650: float = optionalfield(readfunction=float)
  MeanCytoplasm690: float = optionalfield(readfunction=float)
  MeanCytoplasm780: float = optionalfield(readfunction=float)
  TotalNucleusDAPI: float
  TotalNucleus480: float
  TotalNucleus520: float
  TotalNucleus540: float
  TotalNucleus570: float
  TotalNucleus620: float
  TotalNucleus650: float
  TotalNucleus690: float
  TotalNucleus780: float
  TotalMembraneDAPI: float = optionalfield(readfunction=float)
  TotalMembrane480: float = optionalfield(readfunction=float)
  TotalMembrane520: float = optionalfield(readfunction=float)
  TotalMembrane540: float = optionalfield(readfunction=float)
  TotalMembrane570: float = optionalfield(readfunction=float)
  TotalMembrane620: float = optionalfield(readfunction=float)
  TotalMembrane650: float = optionalfield(readfunction=float)
  TotalMembrane690: float = optionalfield(readfunction=float)
  TotalMembrane780: float = optionalfield(readfunction=float)
  TotalEntireCellDAPI: float
  TotalEntireCell480: float
  TotalEntireCell520: float
  TotalEntireCell540: float
  TotalEntireCell570: float
  TotalEntireCell620: float
  TotalEntireCell650: float
  TotalEntireCell690: float
  TotalEntireCell780: float
  TotalCytoplasmDAPI: float = optionalfield(readfunction=float)
  TotalCytoplasm480: float = optionalfield(readfunction=float)
  TotalCytoplasm520: float = optionalfield(readfunction=float)
  TotalCytoplasm540: float = optionalfield(readfunction=float)
  TotalCytoplasm570: float = optionalfield(readfunction=float)
  TotalCytoplasm620: float = optionalfield(readfunction=float)
  TotalCytoplasm650: float = optionalfield(readfunction=float)
  TotalCytoplasm690: float = optionalfield(readfunction=float)
  TotalCytoplasm780: float = optionalfield(readfunction=float)
  ExprPhenotype: int

def MakeClinicalInfo(filename):
  __dateformat = "%m/%d/%Y"

  __nodefault = object()
  def annotationanddefault(fieldname):
    if fieldname in ("REDCapID", "AgeAtCollection", "PackYears"): return int, __nodefault
    if fieldname in ("Date_Death",): return datetime.datetime, datefield(__dateformat, optional=True)
    if "Date" in fieldname: return datetime.datetime, datefield(__dateformat)
    return str, __nodefault

  renamekwargs = {}
  @classmethod
  def transforminitargs(cls, *args, **kwargs):
    kwargs = {renamekwargs.get(kw, kw): kwarg for kw, kwarg in kwargs.items()}
    return super(ClinicalInfo, cls).transforminitargs(*args, **kwargs)

  with open(filename) as f:
    reader = csv.DictReader(f)
    annotations = {}
    defaults = {"transforminitargs": transforminitargs}
    for fieldname in reader.fieldnames:
      if " " in fieldname:
        newname = fieldname.replace(" ", "")
        renamekwargs[fieldname] = newname
        fieldname = newname
      annotations[fieldname], default = annotationanddefault(fieldname)
      if default is not __nodefault:
        defaults[fieldname] = default
  ClinicalInfo = dataclassy.make_dataclass(
    name="ClinicalInfo",
    defaults=defaults,
    fields=annotations,
    bases=(MyDataClass,),
  )
  return ClinicalInfo

class ControlCore(MyDataClass):
  ncore: int
  project: int
  cohort: int
  TMA: int
  cx: int
  cy: int
  Core: str
  Tissue: str

class ControlFlux(MyDataClass):
  project: int
  cohort: int
  core: int
  tma: int
  batch: int
  marker: int
  m1: float
  m2: float

class ControlSample(MyDataClass):
  __dateformat = "%m.%d.%Y"
  Project: int
  Cohort: int
  CtrlID: int
  TMA: int
  Ctrl: int
  Date: datetime.datetime = datefield(__dateformat)
  BatchID: int
  Scan: str
  SlideID: str
