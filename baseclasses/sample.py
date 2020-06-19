import abc, methodtools, pathlib

from ..utilities.misc import dataclass_dc_init, tiffinfo

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
        kwargs["Scan"] = max(int(folder.name.replace("Scan", "")) for folder in (root/kwargs["SlideID"]/"im3").glob("Scan*/"))
      if "BatchID" not in kwargs:
        with open(root/kwargs["SlideID"]/"im3"/f"Scan{kwargs['Scan']}"/"BatchID.txt") as f:
          kwargs["BatchID"] = int(f.read())

    if "SampleID" not in kwargs: kwargs["SampleID"] = 0

    return self.__dc_init__(*args, **kwargs)

  def __bool__(self):
    return bool(self.isGood)

class SampleBase(abc.ABC):
  def __init__(self, root, samp):
    self.root = pathlib.Path(root)
    self.samp = SampleDef(root=root, samp=samp)

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
    componenttifffilename = next(self.componenttiffsfolder.glob(self.SlideID+"*_component_data.tif"))
    return tiffinfo(filename=componenttifffilename)

  @property
  def tiffpscale(self): return self.getcomponenttiffinfo()[0]
  @property
  def tiffwidth(self): return self.getcomponenttiffinfo()[1]
  @property
  def tiffheight(self): return self.getcomponenttiffinfo()[2]
