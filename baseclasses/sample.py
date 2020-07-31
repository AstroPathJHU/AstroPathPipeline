import abc, contextlib, dataclasses, datetime, itertools, jxmlease, logging, methodtools, numpy as np, os, pathlib, re

from ..utilities import units
from ..utilities.misc import dataclass_dc_init, floattoint, tiffinfo
from ..utilities.tableio import readtable, writetable
from .annotationxmlreader import AnnotationXMLReader
from .csvclasses import Constant, RectangleFile
from .logging import getlogger
from .rectangle import Rectangle, rectangleoroverlapfilter
from .overlap import Overlap, RectangleOverlapCollection

@dataclass_dc_init(frozen=True)
class SampleDef:
  SampleID: int
  SlideID: str
  Project: int = None
  Cohort: int = None
  Scan: int = None
  BatchID: int = None
  isGood: int = True

  def __init__(self, *args, root=None, samp=None, **kwargs):
    if samp is not None:
      if isinstance(samp, str):
        if "SlideID" in kwargs:
          raise TypeError("Provided both samp and SlideID")
        else:
          kwargs["SlideID"] = samp
      else:
        if args or kwargs:
          raise TypeError("Have to give either a sample or other arguments, not both.")
        return self.__dc_init__(*args, **kwargs, **{field.name: getattr(samp, field.name) for field in dataclasses.fields(SampleDef)})

    if "SlideID" in kwargs and root is not None:
      root = pathlib.Path(root)
      if "Scan" not in kwargs:
        try:
          kwargs["Scan"] = max(int(folder.name.replace("Scan", "")) for folder in (root/kwargs["SlideID"]/"im3").glob("Scan*/"))
        except ValueError:
          pass
      if "BatchID" not in kwargs and kwargs.get("Scan", None) is not None:
        try:
          with open(root/kwargs["SlideID"]/"im3"/f"Scan{kwargs['Scan']}"/"BatchID.txt") as f:
            kwargs["BatchID"] = int(f.read())
        except FileNotFoundError:
          pass

    if "SampleID" not in kwargs: kwargs["SampleID"] = 0

    return self.__dc_init__(*args, **kwargs)

  def __bool__(self):
    return bool(self.isGood)

class SampleBase(contextlib.ExitStack):
  def __init__(self, root, samp, *, uselogfiles=False, logthreshold=logging.DEBUG):
    self.root = pathlib.Path(root)
    self.samp = SampleDef(root=root, samp=samp)
    if not (self.root/self.SlideID).exists():
      raise IOError(f"{self.root/self.SlideID} does not exist")
    self.logger = getlogger(module=self.logmodule, root=self.root, samp=self.samp, uselogfiles=uselogfiles, threshold=logthreshold)
    super().__init__()

  @property
  def SampleID(self): return self.samp.SampleID
  @property
  def SlideID(self): return self.samp.SlideID
  @property
  def Project(self): return self.samp.Project
  @property
  def Cohort(self): return self.samp.Cohort
  @property
  def Scan(self): return self.samp.Scan
  @property
  def BatchID(self): return self.samp.BatchID
  @property
  def isGood(self): return self.samp.isGood
  def __bool__(self): return bool(self.samp)

  def __str__(self):
    return str(self.mainfolder)

  @property
  def mainfolder(self):
    return self.root/self.SlideID

  @property
  def im3folder(self):
    return self.mainfolder/"im3"

  @property
  def scanfolder(self):
    return self.im3folder/f"Scan{self.Scan}"

  @property
  def componenttiffsfolder(self):
    return self.mainfolder/"inform_data"/"Component_Tiffs"

  def getimageinfofromcomponenttiff(self):
    try:
      componenttifffilename = next(self.componenttiffsfolder.glob(self.SlideID+"*_component_data.tif"))
    except StopIteration:
      raise FileNotFoundError(f"No component tiffs for {self}")
    return tiffinfo(filename=componenttifffilename)

  def getimageinfofromXMLfiles(self):
    with open(self.im3folder/"xml"/(self.SlideID+".Parameters.xml"), "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="/IM3Fragment/D"):
        if node.xml_attrs["name"] == "Shape":
          width, height, nlayers = (int(_) for _ in str(node).split())
        if node.xml_attrs["name"] == "MillimetersPerPixel":
          pscale = 1e-3/float(node)

    try:
      width = units.Distance(pixels=width, pscale=pscale)
      height = units.Distance(pixels=height, pscale=pscale)
    except NameError:
      raise IOError(f'Couldn\'t find Shape and/or MillimetersPerPixel in {self.im3folder/"xml"/(self.SlideID+".Parameters.xml")}')

    return pscale, width, height

  @methodtools.lru_cache()
  @property
  def nlayers(self):
    with open(self.im3folder/"xml"/(self.SlideID+".Parameters.xml"), "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="/IM3Fragment/D"):
        if node.xml_attrs["name"] == "Shape":
          width, height, nlayers = (int(_) for _ in str(node).split())
          return nlayers
      else:
        raise IOError(f'Couldn\'t find Shape in {self.im3folder/"xml"/(self.SlideID+".Parameters.xml")}')

  def getimageinfos(self):
    result = {}
    try:
      result["component tiff"] = self.getimageinfofromcomponenttiff()
    except FileNotFoundError:
      result["component tiff"] = None
    try:
      result["xml files"] = self.getimageinfofromXMLfiles()
    except (FileNotFoundError, IOError):
      result["xml files"] = None
    return result

  @methodtools.lru_cache()
  def getimageinfo(self):
    results = self.getimageinfos()
    result = None
    warnfunction = None
    for k, v in results.items():
      if v is None: continue
      pscale, width, height = v
      if result is None:
        result = resultpscale, resultwidth, resultheight = v
      elif np.allclose(units.microns(result, power=[0, 1, 1], pscale=result[0]), units.microns(v, power=[0, 1, 1], pscale=v[0]), rtol=1e-6):
        continue
      elif np.allclose(units.microns(result, power=[0, 1, 1], pscale=result[0]), units.microns(v, power=[0, 1, 1], pscale=v[0]), rtol=1e-6):
        if warnfunction != self.logger.warningglobal: warnfunction = self.logger.warning
      else:
        warnfunction = self.logger.warningglobal

    if warnfunction is not None:
      fmt = "{:30} {:30} {:30} {:30}"
      warninglines = [
        "Found inconsistent image infos from different sources:",
        fmt.format("source", "pscale", "field width", "field height"),
      ]
      for k, v in results.items():
        if v is None: continue
        warninglines.append(fmt.format(k, *(str(_) for _ in v)))
      for warningline in warninglines:
        warnfunction(warningline)

    if result is None:
      raise FileNotFoundError("Didn't find any of the possible ways of finding image info: "+", ".join(results))

    return result

  @property
  def pscale(self): return self.getimageinfo()[0]
  @property
  def fwidth(self): return self.getimageinfo()[1]
  @property
  def fheight(self): return self.getimageinfo()[2]

  def __enter__(self):
    self.enter_context(self.logger)
    return super().__enter__()

  @abc.abstractproperty
  def logmodule(self):
    "name of the log files for this class (e.g. align)"

class DbloadSampleBase(SampleBase):
  @property
  def dbload(self):
    return self.mainfolder/"dbload"

  def csv(self, csv):
    return self.mainfolder/"dbload"/f"{self.SlideID}_{csv}.csv"
  def readcsv(self, csv, *args, **kwargs):
    return readtable(self.csv(csv), *args, **kwargs)
  def writecsv(self, csv, *args, **kwargs):
    return writetable(self.csv(csv), *args, **kwargs)

  def getimageinfofromconstants(self):
    tmp = self.readcsv("constants", Constant, extrakwargs={"pscale": 1})
    pscale = {_.value for _ in tmp if _.name == "pscale"}.pop()

    constants = self.readcsv("constants", Constant, extrakwargs={"pscale": pscale})
    constantsdict = {constant.name: constant.value for constant in constants}

    fwidth    = constantsdict["fwidth"]
    fheight   = constantsdict["fheight"]
    pscale    = float(constantsdict["pscale"])

    return pscale, fwidth, fheight

  def getimageinfos(self):
    result = super().getimageinfos()

    try:
      result["constants.csv"] = self.getimageinfofromconstants()
    except (FileNotFoundError, KeyError):
      result["constants.csv"] = None

    return result

  @methodtools.lru_cache()
  @property
  def constantsdict(self):
    constants = self.readcsv("constants", Constant, extrakwargs={"pscale": self.pscale})
    return {constant.name: constant.value for constant in constants}

  @property
  def position(self):
    return np.array([self.constantsdict["xposition"], self.constantsdict["yposition"]])
  @property
  def nclip(self):
    return self.constantsdict["nclip"]

class FlatwSampleBase(SampleBase):
  def __init__(self, root, root2, samp, *args, **kwargs):
    super().__init__(root=root, samp=samp, *args, **kwargs)
    self.root2 = pathlib.Path(root2)

  @property
  def root1(self): return self.root

class SampleThatReadsOverlaps(SampleBase):
  rectangletype = Rectangle
  overlaptype = Overlap #can be overridden in subclasses

class ReadRectanglesBase(FlatwSampleBase, SampleThatReadsOverlaps, RectangleOverlapCollection):
  @abc.abstractmethod
  def readallrectangles(self): pass
  @property
  def rectangleextrakwargs(self):
    return {"pscale": self.pscale}
  @abc.abstractmethod
  def readalloverlaps(self): pass
  @property
  def overlapextrakwargs(self):
    return {"pscale": self.pscale, "rectangles": self.rectangles, "nclip": self.nclip}

  def __init__(self, *args, selectrectangles=None, selectoverlaps=None, onlyrectanglesinoverlaps=False, layer=1, readlayerfile=True, **kwargs):
    self.__layer = layer
    self.__readlayerfile = readlayerfile

    super().__init__(*args, **kwargs)

    rectanglefilter = rectangleoroverlapfilter(selectrectangles)
    _overlapfilter = rectangleoroverlapfilter(selectoverlaps)
    overlapfilter = lambda o: _overlapfilter(o) and o.p1 in self.rectangleindices and o.p2 in self.rectangleindices

    self.__rectangles  = self.readallrectangles()
    self.__rectangles = [r for r in self.rectangles if rectanglefilter(r)]
    self.__overlaps  = self.readalloverlaps()
    self.__overlaps = [o for o in self.overlaps if overlapfilter(o)]
    if onlyrectanglesinoverlaps:
      self.__rectangles = [r for r in self.rectangles if self.selectoverlaprectangles(r)]

  @property
  def overlaps(self): return self.__overlaps
  @property
  def rectangles(self): return self.__rectangles

  @property
  def layer(self): return self.__layer

class ReadRectangles(ReadRectanglesBase, DbloadSampleBase):
  def readallrectangles(self):
    return self.readcsv("rect", self.rectangletype, extrakwargs=self.rectangleextrakwargs)
  def readalloverlaps(self):
    return self.readcsv("overlap", self.overlaptype, filter=lambda row: row["p1"] in self.rectangleindices and row["p2"] in self.rectangleindices, extrakwargs=self.overlapextrakwargs)

class XMLLayoutReader(SampleThatReadsOverlaps):
  @methodtools.lru_cache()
  def getlayout(self):
    rectangles, globals, perimeters = self.getXMLplan()
    rectanglefiles = self.getdir()
    maxtimediff = datetime.timedelta(0)
    for r in rectangles:
      rfs = {rf for rf in rectanglefiles if np.all(rf.cxvec == r.cxvec)}
      assert len(rfs) <= 1
      if not rfs:
        cx, cy = units.microns(r.cxvec, pscale=self.pscale)
        raise FileNotFoundError(f"File {self.SlideID}_[{cx},{cy}].im3 (expected from annotations) does not exist")
      rf = rfs.pop()
      maxtimediff = max(maxtimediff, abs(rf.t-r.t))
    if maxtimediff >= datetime.timedelta(seconds=5):
      self.logger.warning(f"Biggest time difference between annotation and file mtime is {maxtimediff}")
    rectangles.sort(key=lambda x: x.t)
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i
    if not rectangles:
      raise ValueError("No layout annotations")
    return rectangles, globals

  @methodtools.lru_cache()
  def getXMLplan(self):
    xmlfile = self.scanfolder/(self.SlideID+"_"+self.scanfolder.name+"_annotations.xml")
    reader = AnnotationXMLReader(xmlfile, pscale=self.pscale)

    rectangles = reader.rectangles
    globals = reader.globals
    perimeters = reader.perimeters
    self.fixM2(rectangles)
    self.fixrectanglefilenames(rectangles)

    return rectangles, globals, perimeters

  def fixM2(self, rectangles):
    for rectangle in rectangles[:]:
      if "_M2" in rectangle.file:
        duplicates = [r for r in rectangles if r is not rectangle and np.all(r.cxvec == rectangle.cxvec)]
        if not duplicates:
          rectangle.file = rectangle.file.replace("_M2", "")
        for d in duplicates:
          rectangles.remove(d)
        self.logger.warningglobal(f"{rectangle.file} has _M2 in the name.  {len(duplicates)} other duplicate rectangles.")
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i

  def fixrectanglefilenames(self, rectangles):
    for r in rectangles:
      expected = self.SlideID+f"_[{floattoint(units.microns(r.cx, pscale=r.pscale), atol=1e-10):d},{floattoint(units.microns(r.cy, pscale=r.pscale), atol=1e-10):d}].im3"
      actual = r.file
      if expected != actual:
        self.logger.warningglobal(f"rectangle at ({r.cx}, {r.cy}) has the wrong filename {actual}.  Changing it to {expected}.")
      r.file = expected

  @methodtools.lru_cache()
  def getdir(self):
    folder = self.scanfolder/"MSI"
    im3s = folder.glob("*.im3")
    result = []
    for im3 in im3s:
      regex = self.SlideID+r"_\[([0-9]+),([0-9]+)\].im3"
      match = re.match(regex, im3.name)
      if not match:
        raise ValueError(f"Unknown im3 filename {im3}, should match {regex}")
      x = units.Distance(microns=int(match.group(1)), pscale=self.pscale)
      y = units.Distance(microns=int(match.group(2)), pscale=self.pscale)
      t = datetime.datetime.fromtimestamp(os.path.getmtime(im3)).astimezone()
      result.append(
        RectangleFile(
          cx=x,
          cy=y,
          t=t,
          pscale=self.pscale,
        )
      )
    return result

  @methodtools.lru_cache()
  def getoverlaps(self):
    overlaps = []
    for r1, r2 in itertools.product(self.rectangles, repeat=2):
      if r1 is r2: continue
      if np.all(abs(r1.cxvec - r2.cxvec) < r1.shape):
        tag = int(np.sign(r1.cx-r2.cx)) + 3*int(np.sign(r1.cy-r2.cy)) + 5
        overlaps.append(
          self.overlaptype(
            n=len(overlaps)+1,
            p1=r1.n,
            p2=r2.n,
            x1=r1.x,
            y1=r1.y,
            x2=r2.x,
            y2=r2.y,
            tag=tag,
            nclip=self.nclip,
            rectangles=(r1, r2),
            pscale=self.pscale,
            readingfromfile=False,
          )
        )
    return overlaps

class ReadRectanglesFromXML(ReadRectanglesBase, XMLLayoutReader):
  def readallrectangles(self):
    rectangles = self.getlayout()[0]
    return [self.rectangletype(rectangle=r, readingfromfile=False, **self.rectangleextrakwargs) for r in rectangles]
  def readalloverlaps(self):
    return self.getoverlaps()
