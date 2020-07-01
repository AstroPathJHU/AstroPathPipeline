import abc, contextlib, dataclasses, logging, methodtools, pathlib

from ..utilities.misc import dataclass_dc_init, tiffinfo
from .logging import getlogger

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
      raise IOError(f"{self.root1/self.SlideID} does not exist")
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
  def dbload(self):
    return self.mainfolder/"dbload"

  @property
  def im3folder(self):
    return self.mainfolder/"im3"

  @property
  def scanfolder(self):
    return self.im3folder/f"Scan{self.Scan}"

  @property
  def componenttiffsfolder(self):
    return self.mainfolder/"inform_data"/"Component_Tiffs"

  @methodtools.lru_cache()
  def getcomponenttiffinfo(self):
    try:
      componenttifffilename = next(self.componenttiffsfolder.glob(self.SlideID+"*_component_data.tif"))
    except StopIteration:
      raise OSError(f"No component tiffs for {self}")
    return tiffinfo(filename=componenttifffilename)

  @property
  def tiffpscale(self): return self.getcomponenttiffinfo()[0]
  @property
  def tiffwidth(self): return self.getcomponenttiffinfo()[1]
  @property
  def tiffheight(self): return self.getcomponenttiffinfo()[2]

  def __enter__(self):
    self.enter_context(self.logger)
    return super().__enter__()

  @abc.abstractproperty
  def logmodule(self):
    "name of the log files for this class (e.g. align)"

class FlatwSampleBase(SampleBase):
  def __init__(self, root, root2, samp, *args, **kwargs):
    super().__init__(root=root, samp=samp, *args, **kwargs)
    self.root2 = pathlib.Path(root2)

  @property
  def root1(self): return self.root
