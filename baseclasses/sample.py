import abc, contextlib, dataclasses, logging, methodtools, numpy as np, pathlib

from ..utilities import units
from ..utilities.misc import dataclass_dc_init, memmapcontext, tiffinfo
from ..utilities.tableio import readtable, writetable
from .csvclasses import Constant
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
      raise OSError(f"No component tiffs for {self}")
    return tiffinfo(filename=componenttifffilename)

  @methodtools.lru_cache()
  def getimageinfo(self):
    return self.getimageinfofromcomponenttiff()

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

  def getimageinfofromconstants(self, *, pscale=None):
    if pscale is None:
      tmp = self.readcsv("constants", Constant, extrakwargs={"pscale": 1})
      pscale = {_.value for _ in tmp if _.name == "pscale"}.pop()
    constants = self.readcsv("constants", Constant, extrakwargs={"pscale": pscale})
    constantsdict = {constant.name: constant.value for constant in constants}

    fwidth    = constantsdict["fwidth"]
    fheight   = constantsdict["fheight"]
    pscale    = float(constantsdict["pscale"])

    return pscale, fwidth, fheight

  @methodtools.lru_cache()
  def getimageinfo(self):
    try:
      tiffpscale, tiffwidth, tiffheight = self.getimageinfofromcomponenttiff()
    except OSError:
      tiffpscale = tiffwidth = tiffheight = None

    try:
      constpscale, constwidth, constheight = self.getimageinfofromconstants(pscale=tiffpscale)
    except FileNotFoundError:
      constpscale = constwidth = constheight = None

    if tiffpscale is constpscale is None:
      raise FileNotFoundError("No component tiff and no constants.csv, can't get image info")

    if constpscale is None: return tiffpscale, tiffwidth, tiffheight

    if tiffpscale is None:
      self.logger.warningglobal("couldn't find a component tiff: trusting image size and pscale from constants.csv")
      return constpscale, constwidth, constheight

    #both are not None
    if (tiffwidth, tiffheight) != (constwidth, constheight):
      self.logger.warningglobal(f"component tiff has size {tiffwidth} {tiffheight} which is different from {constwidth} {constheight} (in constants.csv)")
    if constpscale != tiffpscale:
      if np.isclose(constpscale, tiffpscale, rtol=1e-6):
        warnfunction = self.logger.warning
      else:
        warnfunction = self.logger.warningglobal
      warnfunction(f"component tiff has pscale {tiffpscale} which is different from {constpscale} (in constants.csv)")

    return tiffpscale, tiffwidth, tiffheight

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

class ReadRectangles(FlatwSampleBase, DbloadSampleBase, RectangleOverlapCollection):
  overlaptype = Overlap #can be overridden in subclasses

  def __init__(self, *args, selectrectangles=None, selectoverlaps=None, onlyrectanglesinoverlaps=False, **kwargs):
    super().__init__(*args, **kwargs)

    rectanglefilter = rectangleoroverlapfilter(selectrectangles)
    _overlapfilter = rectangleoroverlapfilter(selectoverlaps)
    overlapfilter = lambda o: _overlapfilter(o) and o.p1 in self.rectangleindices and o.p2 in self.rectangleindices

    self.__rectangles  = self.readcsv("rect", Rectangle, extrakwargs={"pscale": self.pscale})
    self.__rectangles = [r for r in self.rectangles if rectanglefilter(r)]
    self.__overlaps  = self.readcsv("overlap", self.overlaptype, filter=lambda row: row["p1"] in self.rectangleindices and row["p2"] in self.rectangleindices, extrakwargs={"pscale": self.pscale, "layer": self.layer, "rectangles": self.rectangles, "nclip": self.nclip})
    self.__overlaps = [o for o in self.overlaps if overlapfilter(o)]
    if onlyrectanglesinoverlaps:
      self.__rectangles = [r for r in self.rectangles if self.selectoverlaprectangles(r)]
    
  def getrawlayers(self, filetype):
    self.logger.info("getrawlayers")
    if filetype=="flatWarpDAPI" :
      ext = f".fw{self.layer:02d}"
    elif filetype=="camWarpDAPI" :
      ext = f".camWarp_layer{self.layer:02d}"
    else :
      raise ValueError(f"requested file type {filetype} not recognized by getrawlayers")
    path = self.root2/self.SlideID

    rawimages = np.ndarray(shape=(len(self.rectangles), units.pixels(self.fheight, pscale=self.pscale), units.pixels(self.fwidth, pscale=self.pscale)), dtype=np.uint16)

    if not self.rectangles:
      raise IOError(1, "didn't find any rows in the rectangles table for "+self.SlideID)

    for i, rectangle in enumerate(self.rectangles):
      filename = path/rectangle.file.replace(".im3", ext)
      self.logger.info(f"loading rectangle {i+1}/{len(self.rectangles)}")
      with open(filename, "rb") as f:
        #use fortran order, like matlab!
        with memmapcontext(
          f,
          dtype=np.uint16,
          shape=(units.pixels(self.fheight, pscale=self.pscale), units.pixels(self.fwidth, pscale=self.pscale)),
          order="F",
          mode="r"
        ) as memmap:
          rawimages[i] = memmap

    return rawimages

  @property
  def overlaps(self): return self.__overlaps
  @property
  def rectangles(self): return self.__rectangles

  @abc.abstractproperty
  def layer(self): pass
