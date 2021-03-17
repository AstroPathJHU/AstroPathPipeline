import abc, argparse, pathlib, re
from ..utilities import units
from ..utilities.tableio import readtable
from .logging import getlogger
from .sample import SampleDef
from .workflowdependency import SampleRunStatus, ThingWithRoots

class Cohort(ThingWithRoots):
  """
  Base class for a cohort - a bunch of samples that can be run in a loop

  root: the root path of the Cohort, i.e. Clinical_Specimen_*
  filters: functions that are called on each sample to filter it
           if any filter returns False, the sample is skipped
           (default: [])
  debug: raise error messages instead of logging them and continuing
         (default: False)
  uselogfiles, logroot: these arguments are passed to the logger
  """
  def __init__(self, root, *, slideidfilters=[], samplefilters=[], debug=False, uselogfiles=True, logroot=None, xmlfolders=[], version_requirement=None):
    super().__init__()
    self.root = pathlib.Path(root)
    if logroot is None: logroot = root
    self.logroot = pathlib.Path(logroot)
    self.slideidfilters = slideidfilters
    self.samplefilters = samplefilters
    self.debug = debug
    self.uselogfiles = uselogfiles
    self.xmlfolders = xmlfolders

    if version_requirement is None:
      version_requirement = "commit" if self.uselogfiles else "any"

    if version_requirement == "any":
      checkdate = checktag = False
    elif version_requirement == "commit":
      checkdate = True
      checktag = False
    elif version_requirement == "tag":
      checkdate = checktag = True
    else:
      assert False, version_requirement

    from ..utilities.version import astropathversionmatch, env_var_no_git
    if checkdate:
      if env_var_no_git:
        raise RuntimeError("Cannot run if environment variable _ASTROPATH_VERSION_NO_GIT is set unless you set --allow-local-edits")
      if astropathversionmatch.group("date"):
        raise ValueError("Cannot run with local edits to git unless you set --allow-local-edits")
    if checktag:
      if astropathversionmatch.group("dev"):
        raise ValueError("Specified --no-dev-version, but the current version is a dev version")

  def __iter__(self):
    """
    Iterate over the sample's sampledef.csv file.
    It yields all the good samples (as defined by the isGood column)
    that pass the filters.
    """
    for samp in readtable(self.root/"sampledef.csv", SampleDef):
      if not samp: continue
      try:
        if not all(filter(self, samp) for filter in self.slideidfilters): continue
      except:
        with self.getlogger(samp):
          raise
      yield samp

  @abc.abstractmethod
  def runsample(self, sample, **kwargs):
    "actually run whatever is supposed to be run on the sample"

  @property
  @abc.abstractmethod
  def sampleclass(self):
    "What type of samples to create"

  def initiatesample(self, samp):
    "Create a Sample object (subclass of SampleBase) from SampleDef samp to run on"
    return self.sampleclass(samp=samp, **self.initiatesamplekwargs)

  @property
  def initiatesamplekwargs(self):
    "Keyword arguments to pass to the sample class"
    return {"root": self.root, "reraiseexceptions": self.debug, "uselogfiles": self.uselogfiles, "logroot": self.logroot, "xmlfolders": self.xmlfolders}

  @classmethod
  def logmodule(cls):
    "name of the log files for this class (e.g. align)"
    return cls.sampleclass.logmodule()

  @property
  def rootnames(self):
    return {*super().rootnames, "root", "logroot"}
  @property
  def workflowkwargs(self):
    return self.rootkwargs

  def getlogger(self, samp):
    return getlogger(module=self.logmodule(), root=self.logroot, samp=samp, uselogfiles=self.uselogfiles, reraiseexceptions=self.debug)

  def run(self, **kwargs):
    """
    Run the cohort by iterating over the samples and calling runsample on each.
    """
    for samp in self:
      try:
        sample = self.initiatesample(samp)
      except:
        #enter the logger here to log exceptions in __init__ of the sample
        with self.getlogger(samp):
          raise
      else:
        if not all(filter(self, sample) for filter in self.samplefilters):
          continue
        if sample.logmodule() != self.logmodule():
          raise ValueError(f"Wrong logmodule: {self.logmodule()} != {sample.logmodule()}")
        self.processsample(sample, **kwargs)

  def processsample(self, sample, **kwargs):
    with sample:
      self.runsample(sample, **kwargs)

  @property
  def dryrunheader(self):
    return "would run the following samples:"
  def dryrun(self, **kwargs):
    """
    Print which samples would be run if you run the cohort
    """
    print(self.dryrunheader)
    for samp in self: print(samp)

  @classmethod
  def argumentparserhelpmessage(cls):
    return cls.__doc__

  @classmethod
  def makeargumentparser(cls):
    """
    Create an argument parser to run this cohort on the command line
    """
    p = argparse.ArgumentParser(description=cls.argumentparserhelpmessage())
    p.add_argument("root", type=pathlib.Path, help="The Clinical_Specimen folder where sample data is stored")
    p.add_argument("--debug", action="store_true", help="exit on errors, instead of logging them and continuing")
    p.add_argument("--sampleregex", type=re.compile, help="only run on SlideIDs that match this regex")
    p.add_argument("--units", choices=("safe", "fast"), default="fast", help="unit implementation (default: fast; safe is only needed for debugging code)")
    p.add_argument("--dry-run", action="store_true", help="print the sample ids that would be run and exit")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--logroot", type=pathlib.Path, help="root location where the log files are stored (default: same as root)")
    g.add_argument("--no-log", action="store_true", help="do not write to log files")
    p.add_argument("--xmlfolder", type=pathlib.Path, action="append", help="additional folders to look for xml metadata", default=[], dest="xmlfolders")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--no-dev-version", help="refuse to run unless the package version is tagged", action="store_const", const="tag", dest="version_requirement")
    g.add_argument("--allow-dev-version", help="ok to run even if the package is at a dev version (default if using a log file)", action="store_const", const="commit", dest="version_requirement")
    g.add_argument("--allow-local-edits", help="ok to run even if there are local edits on top of the git commit (default if not writing to a log file)", action="store_const", const="any", dest="version_requirement")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    """
    Get the kwargs to be passed to the cohort constructor
    from the parsed arguments
    """
    dct = parsed_args_dict
    kwargs = {
      "root": dct.pop("root"),
      "debug": dct.pop("debug"),
      "logroot": dct.pop("logroot"),
      "uselogfiles": not dct.pop("no_log"),
      "xmlfolders": dct.pop("xmlfolders"),
      "version_requirement": dct.pop("version_requirement"),
      "slideidfilters": [],
      "samplefilters": [],
    }
    if kwargs["logroot"] is None: kwargs["logroot"] = kwargs["root"]
    regex = dct.pop("sampleregex")
    if regex is not None:
      kwargs["slideidfilters"].append(lambda self, sample: regex.match(sample.SlideID))
    return kwargs

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    """
    Get the keyword arguments to be passed to cohort.run() from the parsed arguments
    """
    kwargs = {}
    return kwargs

  @classmethod
  def runfromargumentparser(cls, args=None):
    """
    Main function to run the cohort from command line arguments.
    This function can be called in __main__
    """
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
      return cohort

class Im3Cohort(Cohort):
  """
  Base class for any cohort that uses im3 files
  root2: the location of the sharded im3s
  """
  def __init__(self, root, root2, *args, **kwargs):
    super().__init__(root=root, *args, **kwargs)
    self.root2 = pathlib.Path(root2)

  @property
  def root1(self): return self.root

  @property
  def rootnames(self):
    return {*super().rootnames, "root2"}

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "root2": self.root2}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("root2", type=pathlib.Path, help="root location of sharded im3 files")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "root2": parsed_args_dict.pop("root2"),
    }

class DbloadCohort(Cohort):
  """
  Base class for any cohort that uses the dbload folder
  dbloadroot: an alternate root to use for the dbload folder instead of root
              (mostly useful for testing)
              (default: same as root)
  """
  def __init__(self, *args, dbloadroot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if dbloadroot is None: dbloadroot = self.root
    self.dbloadroot = pathlib.Path(dbloadroot)

  @property
  def rootnames(self):
    return {*super().rootnames, "dbloadroot"}

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "dbloadroot": self.dbloadroot}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--dbloadroot", type=pathlib.Path, help="root location of dbload folder (default: same as root)")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "dbloadroot": parsed_args_dict.pop("dbloadroot"),
    }

class ZoomFolderCohort(Cohort):
  """
  Base class for any cohort that uses zoom files
  zoomroot: root for the zoom files
  """
  def __init__(self, *args, zoomroot, **kwargs):
    super().__init__(*args, **kwargs)
    self.zoomroot = pathlib.Path(zoomroot)

  @property
  def rootnames(self):
    return {*super().rootnames, "zoomroot"}

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "zoomroot": self.zoomroot}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--zoomroot", type=pathlib.Path, required=True, help="root location of stitched wsi images")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "zoomroot": parsed_args_dict.pop("zoomroot"),
    }

class DeepZoomCohort(Cohort):
  """
  Base class for any cohort that uses deepzoom files
  deepzoomroot: root for the deepzoom files
  """
  def __init__(self, *args, deepzoomroot, **kwargs):
    super().__init__(*args, **kwargs)
    self.deepzoomroot = pathlib.Path(deepzoomroot)

  @property
  def rootnames(self):
    return {*super().rootnames, "deepzoomroot"}

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "deepzoomroot": self.deepzoomroot}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--deepzoomroot", type=pathlib.Path, required=True, help="root location of deepzoom images")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "deepzoomroot": parsed_args_dict.pop("deepzoomroot"),
    }

class MaskCohort(Cohort):
  """
  Base class for any cohort that uses the mask folder
  maskroot: an alternate root to use for the mask folder instead of root
            (default: same as root)
  """
  def __init__(self, *args, maskroot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if maskroot is None: maskroot = self.root
    self.maskroot = pathlib.Path(maskroot)

  @property
  def rootnames(self):
    return {*super().rootnames, "maskroot"}

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "maskroot": self.maskroot}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--maskroot", type=pathlib.Path, help="root location of mask folder (default: same as root)")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "maskroot": parsed_args_dict.pop("maskroot"),
    }

class SelectRectanglesCohort(Cohort):
  """
  Base class for any cohort that allows the user to select rectangles
  selectrectangles: the rectangle filter (on the command line, a list of ids)
  """
  def __init__(self, *args, selectrectangles=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.selectrectangles = selectrectangles

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "selectrectangles": self.selectrectangles}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--selectrectangles", type=int, nargs="*", metavar="HPFID", help="select only certain HPF IDs to run on")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "selectrectangles": parsed_args_dict.pop("selectrectangles"),
    }

class SelectLayersCohort(Cohort):
  """
  Base class for any cohort that allows the user to select layers
  layers: the layers to use
  """
  def __init__(self, *args, layers=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.layers = layers

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "layers": self.layers}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--layers", type=int, nargs="*", metavar="LAYER", help="select only certain layers to run on")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "layers": parsed_args_dict.pop("layers"),
    }

class TempDirCohort(Cohort):
  """
  Base class for any cohort that wants to use a temporary directory
  temproot: the location where the temporary directories should be created
  """
  def __init__(self, *args, temproot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if temproot is not None: temproot = pathlib.Path(temproot)
    self.temproot = temproot

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "temproot": self.temproot}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--temproot", type=pathlib.Path, help="root folder to save temp files in")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "temproot": parsed_args_dict.pop("temproot"),
    }

class GeomFolderCohort(Cohort):
  """
  Base class for a cohort that uses the _cellgeomload.csv files
  geomroot: an alternate root to use for the geom folder instead of root
              (mostly useful for testing)
              (default: same as root)
  """
  def __init__(self, *args, geomroot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if geomroot is None: geomroot = self.root
    self.geomroot = pathlib.Path(geomroot)

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "geomroot": self.geomroot}

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--geomroot", type=pathlib.Path, help="root location of geom folder (default: same as root)")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "geomroot": parsed_args_dict.pop("geomroot"),
    }

  @property
  def rootnames(self): return {"geomroot", *super().rootnames}

class WorkflowCohort(Cohort):
  """
  Base class for a cohort that runs as a workflow:
  it takes input files and produces output files.
  It will check for you that the output files exist.
  Also you can check which samples have already run
  successfully, and you can automatically filter
  to only run the ones that haven't.
  """

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--skip-finished", action="store_true", help="only run samples that have not already run successfully")
    p.add_argument("--dependencies", action="store_true", help="only run samples whose dependencies have finished by checking the logs")
    p.add_argument("--print-errors", action="store_true", help="instead of running samples, print the status of the ones that haven't run, including error messages")
    p.add_argument("--ignore-error", type=re.compile, action="append", dest="ignore_errors", help="for --print-errors, ignore any errors that match this regex")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
    }
    if parsed_args_dict.pop("skip_finished"):
      kwargs["slideidfilters"].append(lambda self, sample: not self.sampleclass.getrunstatus(SlideID=sample.SlideID, **self.workflowkwargs))
    if parsed_args_dict.pop("dependencies"):
      kwargs["slideidfilters"].append(lambda self, sample: all(dependency.getrunstatus(SlideID=sample.SlideID, **self.workflowkwargs) for dependency in self.sampleclass.workflowdependencies()))
    if parsed_args_dict["print_errors"]:
      kwargs["uselogfiles"] = False
    return kwargs

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "print_errors": parsed_args_dict.pop("print_errors"),
      "ignore_errors": parsed_args_dict.pop("ignore_errors"),
    }
    return kwargs

  def processsample(self, sample, *, print_errors, ignore_errors, **kwargs):
    if print_errors:
      if ignore_errors is None: ignore_errors = []
      status = sample.runstatus
      if status: return
      if status.error and any(ignore.search(status.error) for ignore in ignore_errors): return
      print(f"{sample.SlideID} {status}")
    else:
      try:
        missinginputs = [file for file in sample.inputfiles if not file.exists()]
        if missinginputs:
          raise IOError("Not all required input files exist.  Missing files: " + ", ".join(str(_) for _ in missinginputs))
      except:
        print("exception")
        with self.getlogger(sample):
          raise
        return

      super().processsample(sample, **kwargs)

      try:
        status = sample.runstatus
        #we don't want to do anything if there's an error, because that
        #was already logged so no need to log it again and confuse the issue.
        if (status.missingfiles or not status.ended) and status.error is None:
          raise RuntimeError(f"{sample.SlideID} {status}")
      except:
        with self.getlogger(sample):
          raise
