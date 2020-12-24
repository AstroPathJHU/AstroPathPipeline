import abc, argparse, pathlib, re
from ..utilities import units
from ..utilities.tableio import readtable
from .logging import getlogger
from .sample import SampleDef

class Cohort(abc.ABC):
  def __init__(self, root, *, filter=lambda samp: True, debug=False, uselogfiles=True, logroot=None):
    super().__init__()
    self.root = pathlib.Path(root)
    self.filter = filter
    self.debug = debug
    self.uselogfiles = uselogfiles
    self.logroot = logroot

  def __iter__(self):
    for samp in readtable(self.root/"sampledef.csv", SampleDef):
      if not samp: continue
      if not self.filter(samp): continue
      yield samp

  @abc.abstractmethod
  def runsample(self, sample, **kwargs):
    "actually run whatever is supposed to be run on the sample"

  @abc.abstractproperty
  def sampleclass(self): pass

  def initiatesample(self, samp):
    "Create a Sample object (subclass of SampleBase) from SampleDef samp to run on"
    return self.sampleclass(samp=samp, **self.initiatesamplekwargs)

  @property
  def initiatesamplekwargs(self):
    return {"root": self.root, "reraiseexceptions": self.debug, "uselogfiles": self.uselogfiles, "logroot": self.logroot}

  @abc.abstractproperty
  def logmodule(self):
    "name of the log files for this class (e.g. align)"

  def run(self, **kwargs):
    for samp in self:
      with getlogger(module=self.logmodule, root=self.root, samp=samp, uselogfiles=self.uselogfiles, reraiseexceptions=self.debug):  #log exceptions in __init__ of the sample
        sample = self.initiatesample(samp)
        if sample.logmodule != self.logmodule:
          raise ValueError(f"Wrong logmodule: {self.logmodule} != {sample.logmodule}")
        with sample:
          self.runsample(sample, **kwargs)

  @property
  def dryrunheader(self):
    return "would run the following samples:"
  def dryrun(self, **kwargs):
    print(self.dryrunheader)
    for samp in self: print(samp)

  @classmethod
  def makeargumentparser(cls):
    p = argparse.ArgumentParser()
    p.add_argument("root", type=pathlib.Path)
    p.add_argument("--debug", action="store_true")
    cls.makesampleselectionargumentgroup(p)
    p.add_argument("--units", choices=("safe", "fast"), default="fast")
    p.add_argument("--dry-run", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--logroot", type=pathlib.Path)
    g.add_argument("--no-log", action="store_true")
    return p

  @classmethod
  def makesampleselectionargumentgroup(cls, parser):
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--sampleregex", type=re.compile)
    return g

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    dct = parsed_args_dict
    kwargs = {
      "root": dct.pop("root"),
      "debug": dct.pop("debug"),
      "logroot": dct.pop("logroot"),
      "uselogfiles": not dct.pop("no_log"),
    }
    regex = dct.pop("sampleregex")
    if regex is not None:
      kwargs["filter"] = lambda sample: regex.match(sample.SlideID)
    return kwargs

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {}
    return kwargs

  @classmethod
  def runfromargumentparser(cls, args=None):
    p = cls.makeargumentparser()
    args = p.parse_args(args=args)
    argsdict = args.__dict__.copy()
    with units.setup_context(argsdict.pop("units")):
      dryrun = argsdict.pop("dry_run")
      initkwargs = cls.initkwargsfromargumentparser(argsdict)
      runkwargs = cls.runkwargsfromargumentparser(argsdict)
      if argsdict:
        raise TypeError(f"Some command line arguments were not processed:\n{argsdict}")
      cohort = cls(**initkwargs)
      if dryrun:
        cohort.dryrun(**runkwargs)
      else:
        cohort.run(**runkwargs)

class FlatwCohort(Cohort):
  def __init__(self, root, root2, *args, **kwargs):
    super().__init__(root=root, *args, **kwargs)
    self.root2 = pathlib.Path(root2)

  @property
  def root1(self): return self.root

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "root2": self.root2}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("root2", type=pathlib.Path)
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "root2": parsed_args_dict.pop("root2"),
    }

class DbloadCohort(Cohort):
  def __init__(self, *args, dbloadroot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if dbloadroot is None: dbloadroot = self.root
    self.dbloadroot = pathlib.Path(dbloadroot)

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "dbloadroot": self.dbloadroot}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--dbloadroot", type=pathlib.Path)
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "dbloadroot": parsed_args_dict.pop("dbloadroot"),
    }

class ZoomCohort(Cohort):
  def __init__(self, *args, zoomroot, **kwargs):
    super().__init__(*args, **kwargs)
    self.zoomroot = pathlib.Path(zoomroot)

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "zoomroot": self.zoomroot}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--zoomroot", type=pathlib.Path, required=True)
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "zoomroot": parsed_args_dict.pop("zoomroot"),
    }
