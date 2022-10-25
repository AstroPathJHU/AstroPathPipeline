import abc, dataclassy, pathlib
from ..utilities.config import CONST as UNIV_CONST
from ..utilities.dataclasses import MetaDataAnnotation, MyDataClassFrozen
from ..utilities.tableio import boolasintfield, readtable, writetable

class SampleDefBase(MyDataClassFrozen):
  """
  """
  @abc.abstractmethod
  def SlideID(self): pass
  @abc.abstractmethod
  def Project(self): pass
  @abc.abstractmethod
  def Cohort(self): pass
  @abc.abstractmethod
  def Scan(self): pass
  @abc.abstractmethod
  def BatchID(self): pass

  @property
  @abc.abstractmethod
  def isGood(self): pass

  def __post_init__(self, *args, **kwargs):
    if self.SlideID is None:
      raise TypeError("Have to give a non-None SlideID to SampleDef")
    super().__post_init__(*args, **kwargs)

  @classmethod
  def initargsfromsampledefcsv(cls, *args, root=None, **kwargs):
    if "SlideID" in kwargs and root is not None:
      root = pathlib.Path(root)
      try:
        cohorttable = readtable(cls.sampledefcsv(root=root), cls)
      except (IOError, EOFError):
        pass
      else:
        for row in cohorttable:
          if row.SlideID == kwargs["SlideID"]:
            return cls.transforminitargs(root=root, samp=row)

    return args, kwargs

  @classmethod
  def transforminitargs(cls, *args, root=None, samp=None, **kwargs):
    if samp is not None:
      if isinstance(samp, str):
        if "SlideID" in kwargs:
          raise TypeError("Provided both samp and SlideID")
        else:
          kwargs["SlideID"] = samp
      else:
        newkwargs = {field: getattr(samp, field) for field in set(dataclassy.fields(type(samp))) & set(dataclassy.fields(cls))}
        duplicates = set(newkwargs.keys()) & set(kwargs.keys())
        for dup in duplicates:
          if kwargs[dup] is None:
            kwargs[dup] = newkwargs[dup]
          if kwargs[dup] == newkwargs[dup] or newkwargs[dup] is None:
            del newkwargs[dup]
        duplicates = set(newkwargs.keys()) & set(kwargs.keys())
        if duplicates:
          raise TypeError(f"Provided {', '.join(duplicates)} multiple times, explicitly and within samp")
        kwargs.update(newkwargs)
        if isinstance(samp, SampleDef):
          return super().transforminitargs(*args, **kwargs)

    args, kwargs = cls.initargsfromsampledefcsv(*args, root=root, **kwargs)

    if "SlideID" in kwargs and root is not None:
      if "Scan" not in kwargs:
        try:
          kwargs["Scan"] = max(int(folder.name.replace("Scan", "")) for folder in (root/kwargs["SlideID"]/UNIV_CONST.IM3_DIR_NAME).glob("Scan*/"))
        except ValueError:
          pass
      if "BatchID" not in kwargs and kwargs.get("Scan", None) is not None:
        try:
          with open(root/kwargs["SlideID"]/UNIV_CONST.IM3_DIR_NAME/f"Scan{kwargs['Scan']}"/"BatchID.txt") as f:
            kwargs["BatchID"] = int(f.read())
        except FileNotFoundError:
          pass

    return super().transforminitargs(*args, **kwargs)

class SampleDef(SampleDefBase):
  """
  The sample definition from sampledef.csv in the cohort folder.
  To construct it, you can give all the arguments, or you can give
  SlideID and leave out some of the others.  If you give a root,
  it will try to figure out the other arguments from there.
  """
  SampleID: int = 0
  SlideID: str = None
  Project: int = None
  Cohort: int = None
  Scan: int = None
  BatchID: int = None
  isGood: bool = boolasintfield(True)

  @classmethod
  def initargsfromsampledefcsv(cls, *args, root=None, apidfile=None, **kwargs):
    args, kwargs = super().initargsfromsampledefcsv(*args, root=root, **kwargs)

    Project = kwargs.get("Project", None)
    if Project is None: kwargs.pop("Project", None)

    if "SlideID" in kwargs and root is not None is not Project and apidfile is None:
      apidfile = root/"upkeep_and_progress"/f"AstropathAPIDdef_{Project:d}.csv"
      if not apidfile.exists(): apidfile = None

    if "SlideID" in kwargs and apidfile is not None:
      apidtable = readtable(apidfile, APIDDef)
      for row in apidtable:
        if row.SlideID == kwargs["SlideID"]:
          if "Cohort" not in kwargs:
            kwargs["Cohort"] = row.Cohort
          if "Project" not in kwargs:
            kwargs["Project"] = row.Project
          if "Scan" not in kwargs and row.Scan is not None:
            kwargs["Scan"] = row.Scan
          if "BatchID" not in kwargs:
            kwargs["BatchID"] = row.BatchID
          if "isGood" not in kwargs:
            kwargs["isGood"] = row.isGood

    return args, kwargs

  @classmethod
  def transforminitargs(cls, *args, **kwargs):
    args, kwargs = super().transforminitargs(*args, **kwargs)
    kwargs.pop("apidfile", None)
    return args, kwargs

  def __bool__(self):
    return bool(self.isGood)

  @classmethod
  def sampledefcsv(cls, root):
    return root/"sampledef.csv"

class APIDDef(SampleDefBase):
  SlideID: str = None
  SampleName: str = None
  Project: int = None
  Cohort: int = None
  Scan: int = None
  BatchID: int = None
  isGood: bool = boolasintfield(None)

  def __post_init__(self, **kwargs):
    if self.SlideID is None: raise ValueError("Have to provide SlideID")
    if self.SampleName is None: raise ValueError("Have to provide SampleName")
    if self.Project is None: raise ValueError("Have to provide Project")
    if self.Cohort is None: raise ValueError("Have to provide Cohort")
    if self.BatchID is None: raise ValueError("Have to provide BatchID")
    if self.isGood is None: raise ValueError("Have to provide isGood")
    return super().__post_init__(**kwargs)

  def __bool__(self):
    return bool(self.isGood)

class ControlTMASampleDef(SampleDefBase):
  Project: int
  Cohort: int
  CtrlID: int
  TMA: int
  Ctrl: int
  Date: str
  BatchID: str
  Scan: str
  SlideID: str
  @property
  def isGood(self): return True

class MetadataSummary(MyDataClassFrozen):
  """
  helper dataclass for some common sample metadata information
  """
  slideID         : str
  project         : int
  cohort          : int
  microscope_name : str
  mindate         : str
  maxdate         : str

from .cohort import RunCohortBase

class MakeSampleDef(RunCohortBase):
  def __init__(self, *args, sampledefoutput=None, inputfile, isapid=False, **kwargs):
    super().__init__(*args, **kwargs)
    if sampledefoutput is None:
      sampledefoutput = self.root/"sampledef.csv"
    self.sampledefoutput = pathlib.Path(sampledefoutput)
    self.inputfile = pathlib.Path(inputfile)
    self.isapid = isapid

  def run(self, firstsampleid):
    inputs = sorted([_ for _ in readtable(self.inputfile, APIDDef if self.isapid else SampleDef) if (self.root/_.SlideID).is_dir()], key=lambda x: x.SlideID)
    if self.isapid:
      if firstsampleid is None: raise ValueError("Have to provide firstsampleid if using an APIDDef file")
      sampledefs = [SampleDef(SlideID=apid.SlideID, SampleID=i, root=self.root, apidfile=self.inputfile) for i, apid in enumerate(inputs, start=firstsampleid)]
    else:
      sampledefs = inputs
    writetable(self.sampledefoutput, sampledefs)

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    dct = parsed_args_dict
    apidfile = dct.pop("apidfile")
    sampledeffile = dct.pop("sampledeffile")
    if apidfile is not None is sampledeffile:
      inputfile = apidfile
      isapid = True
    elif apidfile is None is not sampledeffile:
      inputfile = sampledeffile
      isapid = False
    else:
      assert False

    return {
      **super().initkwargsfromargumentparser(dct),
      "inputfile": inputfile,
      "isapid": isapid,
      "sampledefoutput": dct.pop("outfile"),
      "uselogfiles": False,
    }

  @classmethod
  def logmodule(cls): return "makesampledef"

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    dct = parsed_args_dict
    return {
      **super().runkwargsfromargumentparser(dct),
      "firstsampleid": dct.pop("first_sample_id"),
    }

  @classmethod
  def makeargumentparser(cls, **kwargs):
    """
    Create an argument parser to run this cohort on the command line
    """
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--first-sample-id", type=int, help="SampleID for the first SlideID (the others are counted sequentially)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apidfile", type=pathlib.Path, help="path to AstropathAPIDdef.csv")
    g.add_argument("--sampledeffile", type=pathlib.Path, help="path to AstroPathSampledef.csv")
    p.add_argument("--outfile", type=pathlib.Path, help="output filename (default: root/sampledef.csv")
    return p

def makesampledef(args=None):
  return MakeSampleDef.runfromargumentparser(args)
