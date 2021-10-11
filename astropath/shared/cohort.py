import abc, datetime, job_lock, pathlib, re
from ..utilities.config import CONST as UNIV_CONST
from ..utilities import units
from ..utilities.tableio import readtable, TableReader, writetable
from .argumentparser import ArgumentParserMoreRoots, DbloadArgumentParser, DeepZoomArgumentParser, GeomFolderArgumentParser, Im3ArgumentParser, MaskArgumentParser, ParallelArgumentParser, RunFromArgumentParser, SelectLayersArgumentParser, SelectRectanglesArgumentParser, TempDirArgumentParser, XMLPolygonReaderArgumentParser, ZoomFolderArgumentParser, ImageCorrectionArgumentParser
from .logging import getlogger
from .rectangle import rectanglefilter
from .workflowdependency import ThingWithRoots, WorkflowDependency

class MetaCohortBase:
  def __init__(self, *args, mainlogroots=[], **kwargs):
    super().__init__(*args, **kwargs)
    self.mainlogroots = [pathlib.Path(_) for _ in mainlogroots if _ is not None]
  def getlogger(self, *, moremainlogroots, **kwargs):
    return getlogger(moremainlogroots=list(moremainlogroots)+self.mainlogroots, **kwargs)
  @property
  @abc.abstractmethod
  def ProjectsAndCohorts(self): pass

class CohortBase(MetaCohortBase, ThingWithRoots):
  """
  Base class for a cohort.  This class doesn't actually run anything
  (for that use Cohort, below).
  """
  def __init__(self, *args, root, sampledefroot=None, logroot=None, uselogfiles=True, reraiseexceptions=False, **kwargs):
    super().__init__(*args, mainlogroots=[logroot]+moremainlogroots, **kwargs)
    self.__root = pathlib.Path(root)
    if logroot is None: logroot = self.__root
    self.__logroot = pathlib.Path(logroot)
    if sampledefroot is None: sampledefroot = self.__root
    self.__sampledefroot = pathlib.Path(sampledefroot)
    self.uselogfiles = uselogfiles
    self.reraiseexceptions = reraiseexceptions

  @property
  def sampledefs(self):
    from .samplemetadata import SampleDef
    return readtable(self.sampledefroot/"sampledef.csv", SampleDef)
  @property
  def SlideIDs(self): return [_.SlideID for _ in self.sampledefs]
  @property
  def Project(self):
    Project, = {_.Project for _ in self.sampledefs}
    return Project
  @property
  def Cohort(self):
    Cohort, = {_.Cohort for _ in self.sampledefs}
    return Cohort

  def globallogger(self):
    from .samplemetadata import SampleDef
    samp = SampleDef(Project=self.Project, Cohort=self.Cohort, SlideID=f"project{self.Project}")
    return self.getlogger(samp, isglobal=True)

  @property
  def logger(self): return self.globallogger()
  @property
  def mainlogs(self): return self.logger.mainlogs

  def globaljoblock(self, corruptfiletimeout=datetime.timedelta(minutes=10), **kwargs):
    lockfiles = [logfile.with_suffix(".lock") for logfile in self.globallogger().mainlogs]
    return job_lock.MultiJobLock(*lockfiles, corruptfiletimeout=corruptfiletimeout, mkdir=True, **kwargs)

  def getlogger(self, samp, *, isglobal=False, **kwargs):
    if isinstance(samp, WorkflowDependency):
      isglobal = isglobal or samp.usegloballogger()
      samp = samp.samp
    return super().getlogger(module=self.logmodule(), root=self.logroot, samp=samp, uselogfiles=self.uselogfiles, reraiseexceptions=self.reraiseexceptions, isglobal=isglobal, **kwargs)

  @classmethod
  @abc.abstractmethod
  def logmodule(cls): pass

  @property
  def root(self): return self.__root
  @property
  def logroot(self): return self.__logroot
  @property
  def sampledefroot(self): return self.__sampledefroot
  @property
  def rootnames(self):
    return {*super().rootnames, "root", "logroot", "sampledefroot"}

class RunCohortBase(CohortBase, RunFromArgumentParser):
  """
  Base class for a cohort that can be run from the command line
  """

  @abc.abstractmethod
  def run(self): pass
  def dryrun(self, **kwargs):
    print("Command line is valid")

  @classmethod
  def defaultunits(cls):
    return "fast_pixels"

  @classmethod
  def makeargumentparser(cls, **kwargs):
    """
    Create an argument parser to run this cohort on the command line
    """
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--sampledefroot", type=pathlib.Path, help="folder to look for sampledef.csv")
    p.add_argument("--dry-run", action="store_true", help="print the sample ids that would be run and exit")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    """
    Get the kwargs to be passed to the cohort constructor
    from the parsed arguments
    """
    dct = parsed_args_dict
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "sampledefroot": dct.pop("sampledefroot"),
    }
    return kwargs

  @classmethod
  def misckwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().misckwargsfromargumentparser(parsed_args_dict),
      "dry_run": parsed_args_dict.pop("dry_run"),
    }
    return kwargs

  @classmethod
  def runfromargsdicts(cls, *, initkwargs, runkwargs, misckwargs):
    """
    Run the cohort from command line arguments.
    """
    with units.setup_context(misckwargs.pop("units")):
      dryrun = misckwargs.pop("dry_run")
      if misckwargs:
        raise TypeError(f"Some miscellaneous kwargs were not processed:\n{misckwargs}")
      if dryrun:
        initkwargs["uselogfiles"] = False
      cohort = cls(**initkwargs)
      if dryrun:
        cohort.dryrun(**runkwargs)
      else:
        cohort.run(**runkwargs)
      return cohort

class Cohort(RunCohortBase, ArgumentParserMoreRoots):
  """
  Base class for a cohort that can be run in a loop

  root: the root path of the Cohort, i.e. Clinical_Specimen_*
  filters: functions that are called on each sample to filter it
           if any filter returns False, the sample is skipped
           (default: [])
  debug: raise error messages instead of logging them and continuing
         (default: False)
  uselogfiles, logroot: these arguments are passed to the logger
  """
  def __init__(self, *args, slideidfilters=[], samplefilters=[], im3root=None, debug=False, informdataroot=None, xmlfolders=[], version_requirement=None, **kwargs):
    super().__init__(*args, reraiseexceptions=debug, **kwargs)
    if im3root is None: im3root = self.root
    self.im3root = pathlib.Path(im3root)
    if informdataroot is None: informdataroot = self.root
    self.informdataroot = pathlib.Path(informdataroot)
    self.slideidfilters = slideidfilters
    self.samplefilters = samplefilters
    self.xmlfolders = xmlfolders

  @property
  def filteredsampledefs(self):
    """
    Iterate over the sample's sampledef.csv file.
    It yields all the good samples (as defined by the isGood column)
    that pass the filters.
    """
    for samp in self.sampledefs:
      if not samp: continue
      try:
        if not all(filter(self, samp) for filter in self.slideidfilters): continue
      except Exception: #don't log KeyboardInterrupt here
        with self.getlogger(samp):
          raise
      yield samp

  @property
  def allsamples(self) :
    for samp in self.sampledefs:
      try:
        sample = self.initiatesample(samp)
        if sample.logmodule() != self.logmodule():
          raise ValueError(f"Wrong logmodule: {self.logmodule()} != {sample.logmodule()}")
        yield sample
      except Exception:
        #enter the logger here to log exceptions in __init__ of the sample
        #but not KeyboardInterrupt
        with self.getlogger(samp):
          raise

  @property
  def samples(self):
    for samp in self.filteredsampledefs:
      try:
        sample = self.initiatesample(samp)
        if sample.logmodule() != self.logmodule():
          raise ValueError(f"Wrong logmodule: {self.logmodule()} != {sample.logmodule()}")
        yield sample
      except Exception:
        #enter the logger here to log exceptions in __init__ of the sample
        #but not KeyboardInterrupt
        with self.getlogger(samp):
          raise

  @property
  def filteredsamples(self):
    for sample in self.samples:
      if not all(filter(self, sample) for filter in self.samplefilters):
        continue
      yield sample

  def runsample(self, sample, **kwargs):
    "actually run whatever is supposed to be run on the sample"
    sample.run(**kwargs)

  @property
  @abc.abstractmethod
  def sampleclass(cls):
    "What type of samples to create"

  def initiatesample(self, samp):
    "Create a Sample object (subclass of SampleBase) from SampleDef samp to run on"
    return self.sampleclass(samp=samp, **self.initiatesamplekwargs)

  @property
  def initiatesamplekwargs(self):
    "Keyword arguments to pass to the sample class"
    return {"root": self.root, "reraiseexceptions": self.reraiseexceptions, "uselogfiles": self.uselogfiles, "logroot": self.logroot, "im3root": self.im3root, "informdataroot": self.informdataroot, "xmlfolders": self.xmlfolders}

  @classmethod
  def logmodule(cls):
    "name of the log files for this class (e.g. align)"
    return cls.sampleclass.logmodule()

  @property
  def rootnames(self):
    return {*super().rootnames, "im3root", "informdataroot"}
  @property
  def workflowkwargs(self):
    return self.rootkwargs

  def run(self, **kwargs):
    """
    Run the cohort by iterating over the samples and calling runsample on each.
    """
    for sample in self.filteredsamples:
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
    for samp in self.filteredsampledefs: print(samp)

  @classmethod
  def defaultunits(cls):
    return cls.sampleclass.defaultunits()

  @classmethod
  def makeargumentparser(cls, **kwargs):
    """
    Create an argument parser to run this cohort on the command line
    """
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--debug", action="store_true", help="exit on errors, instead of logging them and continuing")
    p.add_argument("--sampleregex", type=re.compile, help="only run on SlideIDs that match this regex")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    """
    Get the kwargs to be passed to the cohort constructor
    from the parsed arguments
    """
    dct = parsed_args_dict
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "debug": dct.pop("debug"),
      "slideidfilters": [],
      "samplefilters": [],
    }
    if dct["dry_run"]:
      kwargs["uselogfiles"] = False
    regex = dct.pop("sampleregex")
    if regex is not None:
      kwargs["slideidfilters"].append(lambda self, sample: regex.match(sample.SlideID))
    return kwargs

class Im3Cohort(Cohort, Im3ArgumentParser):
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

class DbloadCohortBase(CohortBase):
  def __init__(self, *args, dbloadroot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if dbloadroot is None: dbloadroot = self.root
    self.dbloadroot = pathlib.Path(dbloadroot)

  @property
  def rootnames(self):
    return {*super().rootnames, "dbloadroot"}

class DbloadCohort(Cohort, DbloadCohortBase, DbloadArgumentParser):
  """
  Base class for any cohort that uses the dbload folder
  dbloadroot: an alternate root to use for the dbload folder instead of root
              (mostly useful for testing)
              (default: same as root)
  """

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "dbloadroot": self.dbloadroot}

class GlobalDbloadCohortBase(DbloadCohortBase, TableReader):
  @property
  def dbload(self):
    return self.dbloadroot/UNIV_CONST.DBLOAD_DIR_NAME
  def csv(self, csv):
    return self.dbload/f"project{self.Project}_{csv}.csv"
  def readcsv(self, csv, *args, **kwargs):
    return self.readtable(self.csv(csv), *args, **kwargs)
  def writecsv(self, csv, *args, **kwargs):
    return writetable(self.csv(csv), *args, logger=self.logger, **kwargs)

class GlobalDbloadCohort(GlobalDbloadCohortBase, DbloadCohort):
  pass

class ZoomFolderCohort(Cohort, ZoomFolderArgumentParser):
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

class DeepZoomCohort(Cohort, DeepZoomArgumentParser):
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

class MaskCohort(Cohort, MaskArgumentParser):
  """
  Base class for any cohort that uses the mask folder
  maskroot: an alternate root to use for the mask folder instead of root
            (default: same as root)
  """
  def __init__(self, *args, maskroot=None, maskfilesuffix=None, **kwargs):
    super().__init__(*args, **kwargs)
    if maskroot is None: maskroot = self.im3root
    self.maskroot = pathlib.Path(maskroot)
    if maskfilesuffix is None: maskfilesuffix = self.defaultmaskfilesuffix
    self.maskfilesuffix = maskfilesuffix

  @property
  def rootnames(self):
    return {*super().rootnames, "maskroot"}
  @property
  def workflowkwargs(self):
    return {
      **super().workflowkwargs,
      "maskfilesuffix": self.maskfilesuffix,
    }

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "maskroot": self.maskroot, "maskfilesuffix": self.maskfilesuffix}

class SelectRectanglesCohort(Cohort, SelectRectanglesArgumentParser):
  """
  Base class for any cohort that allows the user to select rectangles
  selectrectangles: the rectangle filter (on the command line, a list of ids)
  """
  def __init__(self, *args, selectrectangles=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.selectrectangles = rectanglefilter(selectrectangles)

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "selectrectangles": self.selectrectangles}

  @property
  def workflowkwargs(self):
    return {
      **super().workflowkwargs,
      "selectrectangles": self.selectrectangles,
    }

class SelectLayersCohort(Cohort, SelectLayersArgumentParser):
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

class TempDirCohort(Cohort, TempDirArgumentParser):
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

class GeomFolderCohort(Cohort, GeomFolderArgumentParser):
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

  @property
  def rootnames(self): return {"geomroot", *super().rootnames}

class PhenotypeFolderCohort(Cohort):
  """
  Base class for a cohort that uses the _cleaned_phenotype_table.csv files
  phenotyperoot: an alternate root to use for the phenotype folder instead of root
              (mostly useful for testing)
              (default: same as root)
  """
  def __init__(self, *args, phenotyperoot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if phenotyperoot is None: phenotyperoot = self.informdataroot
    self.phenotyperoot = pathlib.Path(phenotyperoot)

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "phenotyperoot": self.phenotyperoot}

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--phenotyperoot", type=pathlib.Path, help="root location of phenotype folder (default: same as root)")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "phenotyperoot": parsed_args_dict.pop("phenotyperoot"),
    }

  @property
  def rootnames(self): return {"phenotyperoot", *super().rootnames}

class ParallelCohort(Cohort, ParallelArgumentParser):
  def __init__(self, *args, njobs, **kwargs):
    self.__njobs = njobs
    super().__init__(*args, **kwargs)
  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "njobs": self.__njobs,
    }

class XMLPolygonReaderCohort(Cohort, XMLPolygonReaderArgumentParser):
  def __init__(self, *args, annotationsynonyms=None, reorderannotations=False, **kwargs):
    self.__annotationsynonyms = annotationsynonyms
    self.__reorderannotations = reorderannotations
    super().__init__(*args, **kwargs)
  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "annotationsynonyms": self.__annotationsynonyms,
      "reorderannotations": self.__reorderannotations,
    }

class CorrectedImageCohort(Im3Cohort,ImageCorrectionArgumentParser) :
  """
  Class for a cohort that uses corrected im3 images as its sample rectangles
  """
  def __init__(self,*args,et_offset_file,skip_et_corrections,flatfield_file,warping_file,**kwargs) :
    super().__init__(*args,**kwargs)
    self.__et_offset_file = et_offset_file
    self.__skip_et_corrections = skip_et_corrections
    self.__flatfield_file = flatfield_file
    self.__warping_file = warping_file
  @property
  def initiatesamplekwargs(self) :
    return {
      **super().initiatesamplekwargs,
      'et_offset_file': self.__et_offset_file,
      'skip_et_corrections':self.__skip_et_corrections,
      'flatfield_file': self.__flatfield_file,
      'warping_file': self.__warping_file
    }

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
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rerun-error", type=re.compile, action="append", dest="rerun_errors", help="rerun only samples with an error that matches this regex")
    g.add_argument("--rerun-finished", action="store_false", dest="skip_finished", help="rerun samples that have already run successfully")
    p.add_argument("--ignore-dependencies", action="store_false", dest="dependencies", help="try (and probably fail) to run samples whose dependencies have not yet finished")
    p.add_argument("--print-errors", action="store_true", help="instead of running samples, print the status of the ones that haven't run, including error messages")
    p.add_argument("--ignore-error", type=re.compile, action="append", dest="ignore_errors", help="for --print-errors, ignore any errors that match this regex")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
    }
    if parsed_args_dict["print_errors"]:
      kwargs["uselogfiles"] = False
      parsed_args_dict["skip_finished"] = parsed_args_dict["dependencies"] = False

    dependencies = parsed_args_dict.pop("dependencies")
    skip_finished = parsed_args_dict.pop("skip_finished")
    rerun_errors = parsed_args_dict.pop("rerun_errors")

    def filter(runstatus, dependencyrunstatuses):
      if rerun_errors and not any(errorregex.search(runstatus.error) for errorregex in rerun_errors):
        runstatus = True

      if not skip_finished and not dependencies:
        return True

      elif skip_finished and not dependencies:
        return not runstatus

      elif dependencies and not skip_finished:
        for dependencyrunstatus in dependencyrunstatuses:
          if not dependencyrunstatus: return False
        return True

      elif dependencies and skip_finished:
        for dependencyrunstatus in dependencyrunstatuses:
          if not dependencyrunstatus: return False
          if runstatus and runstatus.started < dependencyrunstatus.ended:
            runstatus = False #it's as if this step hasn't run
        return not runstatus
      else:
        assert False

    def slideidfilter(self, sample):
      return filter(
        runstatus=self.sampleclass.getrunstatus(SlideID=sample.SlideID, **self.workflowkwargs),
        dependencyrunstatuses=[
          dependency.getrunstatus(SlideID=sample.SlideID, **self.workflowkwargs)
          for dependency in self.sampleclass.workflowdependencyclasses()
        ],
      )
    kwargs["slideidfilters"].append(slideidfilter)

    def samplefilter(self, sample):
      return filter(
        runstatus=sample.runstatus(),
        dependencyrunstatuses=[
          dependency.getrunstatus(SlideID=SlideID, **self.workflowkwargs)
          for dependency, SlideID in sample.workflowdependencies()
        ],
      )
    kwargs["samplefilters"].append(samplefilter)

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
      status = sample.runstatus(**kwargs)
      if status: return
      if status.error and any(ignore.search(status.error) for ignore in ignore_errors): return
      print(f"{sample.SlideID} {status}")
    else:
      with sample.joblock() as lock:
        if not lock: return
        try:
          missinginputs = [file for file in sample.inputfiles(**kwargs) if not file.exists()]
          if missinginputs:
            raise IOError("Not all required input files exist.  Missing files: " + ", ".join(str(_) for _ in missinginputs))
        except Exception: #don't log KeyboardInterrupt here
          with sample:
            raise
          return

        with self.getlogger(sample):
          super().processsample(sample, **kwargs)

          status = sample.runstatus(**kwargs)
          #we don't want to do anything if there's an error, because that
          #was already logged so no need to log it again and confuse the issue.
          if status.missingfiles and status.error is None:
            status.ended = True #to get the missing files in the __str__
            raise RuntimeError(f"{sample.logger.SlideID} {status}")
