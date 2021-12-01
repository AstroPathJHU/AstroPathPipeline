import abc, contextlib, datetime, job_lock, logging, pathlib, re
from ..utilities.config import CONST as UNIV_CONST
from ..utilities import units
from ..utilities.tableio import readtable, TableReader, writetable
from ..utilities.version.git import thisrepo
from .argumentparser import ArgumentParserMoreRoots, DbloadArgumentParser, DeepZoomArgumentParser, GeomFolderArgumentParser, Im3ArgumentParser, MaskArgumentParser, ParallelArgumentParser, RunFromArgumentParser, SelectLayersArgumentParser, SelectRectanglesArgumentParser, TempDirArgumentParser, XMLPolygonReaderArgumentParser, ZoomFolderArgumentParser, ImageCorrectionArgumentParser
from .logging import getlogger
from .rectangle import rectanglefilter
from .workflowdependency import ThingWithRoots, WorkflowDependency

class CohortBase(ThingWithRoots):
  """
  Base class for a cohort.  This class doesn't actually run anything
  (for that use Cohort, below).
  """
  def __init__(self, *args, root, sampledefroot=None, logroot=None, uselogfiles=True, reraiseexceptions=False, moremainlogroots=[], skipstartfinish=False, printthreshold=logging.DEBUG, **kwargs):
    super().__init__(*args, **kwargs)
    self.__root = pathlib.Path(root)
    if logroot is None: logroot = self.__root
    self.__logroot = pathlib.Path(logroot)
    if sampledefroot is None: sampledefroot = self.__root
    self.__sampledefroot = pathlib.Path(sampledefroot)
    self.uselogfiles = uselogfiles
    self.reraiseexceptions = reraiseexceptions
    self.moremainlogroots = moremainlogroots
    self.skipstartfinish = skipstartfinish
    self.printthreshold = printthreshold

  def sampledefs(self, **kwargs):
    from .samplemetadata import SampleDef
    return readtable(self.sampledefroot/"sampledef.csv", SampleDef)
  @property
  def SlideIDs(self): return [_.SlideID for _ in self.sampledefs()]
  @property
  def Project(self):
    Project, = {_.Project for _ in self.sampledefs()}
    return Project
  @property
  def Cohort(self):
    Cohort, = {_.Cohort for _ in self.sampledefs()}
    return Cohort

  def globallogger(self, **kwargs):
    from .samplemetadata import SampleDef
    samp = SampleDef(Project=self.Project, Cohort=self.Cohort, SlideID=f"project{self.Project}")
    return self.getlogger(samp, isglobal=True, **kwargs)

  def printlogger(self, *args, **kwargs): return self.getlogger(*args, uselogfiles=False, **kwargs)

  @property
  def logger(self): return self.globallogger()
  @property
  def globalprintlogger(self): return self.globallogger(uselogfiles=False)
  @property
  def mainlogs(self): return self.logger.mainlogs

  def globaljoblock(self, corruptfiletimeout=datetime.timedelta(minutes=10), **kwargs):
    lockfiles = [logfile.with_suffix(".lock") for logfile in self.globallogger().mainlogs]
    return job_lock.MultiJobLock(*lockfiles, corruptfiletimeout=corruptfiletimeout, mkdir=True, **kwargs)

  def getlogger(self, samp, *, isglobal=False, uselogfiles=True, **kwargs):
    if isinstance(samp, WorkflowDependency):
      isglobal = isglobal or samp.usegloballogger()
      samp = samp.samp
    return getlogger(module=self.logmodule(), root=self.logroot, samp=samp, uselogfiles=uselogfiles and self.uselogfiles, reraiseexceptions=self.reraiseexceptions, isglobal=isglobal, moremainlogroots=self.moremainlogroots, skipstartfinish=self.skipstartfinish, printthreshold=self.printthreshold, **kwargs)

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
    self.globalprintlogger.critical("Command line is valid")

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

class FilterResult:
  def __init__(self, result, message=None, cleanup=False):
    if isinstance(result, FilterResult):
      result, message, cleanup = result.result, result.message, result.cleanup
    self.result = result
    self.message = message
    self.cleanup = cleanup

  def __bool__(self):
    return bool(self.result)

  def __str__(self):
    return self.message

class SampleFilter:
  def __init__(self, function, truemessage, falsemessage):
    self.function = function
    self.messages = {True: truemessage, False: falsemessage}
  def __call__(self, *args, **kwargs):
    result = FilterResult(self.function(*args, **kwargs))
    if result.message is None:
      result.message = self.messages[bool(result)]
    return result

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
  def __init__(self, *args, slideidfilters=[], samplefilters=[], im3root=None, debug=False, informdataroot=None, xmlfolders=[], **kwargs):
    super().__init__(*args, reraiseexceptions=debug, **kwargs)
    if im3root is None: im3root = self.root
    self.im3root = pathlib.Path(im3root)
    if informdataroot is None: informdataroot = self.root
    self.informdataroot = pathlib.Path(informdataroot)
    self.slideidfilters = slideidfilters
    self.samplefilters = samplefilters
    self.xmlfolders = xmlfolders

  def sampledefswithfilters(self, **kwargs):
    for samp in self.sampledefs():
      if not samp: continue
      try:
        yield samp, [filter(self, samp, **kwargs) for filter in self.slideidfilters]
      except Exception: #don't log KeyboardInterrupt here
        with self.handlesampledeffiltererror(samp, **kwargs):
          raise

  def filteredsampledefswithfilters(self, *, printnotrunning=False, **kwargs):
    """
    Iterate over the sample's sampledef.csv file.
    It yields all the good samples (as defined by the isGood column)
    that pass the filters.
    """
    for samp, filters in self.sampledefswithfilters(**kwargs):
      if all(filters):
        yield samp, filters
      elif printnotrunning:
        logger = self.printlogger(samp)
        logger.info(f"Not running {samp.SlideID}:")
        for filter in filters:
          if not filter:
            logger.info(filter.message)

  def filteredsampledefs(self, **kwargs):
    for samp, filters in self.filteredsampledefswithfilters(**kwargs):
      yield samp

  def allsamples(self, **kwargs) :
    for samp in self.sampledefs(**kwargs):
      try:
        sample = self.initiatesample(samp)
        if sample.logmodule() != self.logmodule():
          raise ValueError(f"Wrong logmodule: {self.logmodule()} != {sample.logmodule()}")
        yield sample
      except Exception:
        #enter the logger here to log exceptions in __init__ of the sample
        #but not KeyboardInterrupt
        with self.handlesampleiniterror(samp, **kwargs):
          raise

  def sampleswithfilters(self, **kwargs):
    for samp, filters in self.filteredsampledefswithfilters(**kwargs):
      try:
        sample = self.initiatesample(samp)
        if sample.logmodule() != self.logmodule():
          raise ValueError(f"Wrong logmodule: {self.logmodule()} != {sample.logmodule()}")
      except Exception:
        #enter the logger here to log exceptions in __init__ of the sample
        #but not KeyboardInterrupt
        with self.handlesampleiniterror(samp, **kwargs):
          raise
      try:
        yield sample, filters + [filter(self, sample, **kwargs) for filter in self.samplefilters]
      except Exception:
        #enter the logger here to log exceptions in __init__ of the sample
        #but not KeyboardInterrupt
        with self.handlesamplefiltererror(samp, **kwargs):
          raise

  def handlesampledeffiltererror(self, samp, **kwargs):
    return self.getlogger(samp)
  def handlesampleiniterror(self, samp, **kwargs):
    return self.getlogger(samp)
  def handlesamplefiltererror(self, samp, **kwargs):
    return self.getlogger(samp)

  def samples(self, **kwargs):
    for sample, filters in self.sampleswithfilters(**kwargs):
      yield sample

  def filteredsamples(self, **kwargs):
    for sample, filters in self.sampleswithfilters(**kwargs):
      if not all(filters):
        continue
      yield sample

  def runsample(self, sample, **kwargs):
    "actually run whatever is supposed to be run on the sample"
    return sample.run(**kwargs)

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
    return {"root": self.root, "reraiseexceptions": self.reraiseexceptions, "uselogfiles": self.uselogfiles, "logroot": self.logroot, "im3root": self.im3root, "informdataroot": self.informdataroot, "xmlfolders": self.xmlfolders, "moremainlogroots": self.moremainlogroots, "skipstartfinish": self.skipstartfinish, "printthreshold": self.printthreshold}

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

  def run(self, *, printnotrunning=True, cleanup=False, **kwargs):
    """
    Run the cohort by iterating over the samples and calling runsample on each.
    """
    result = []
    for sample, filters in self.sampleswithfilters(printnotrunning=printnotrunning, **kwargs):
      if all(filters):
        result.append(self.processsample(sample, cleanup=cleanup or any(_.cleanup for _ in filters), **kwargs))
      elif printnotrunning:
        logger = self.printlogger(sample)
        logger.info(f"Not running {sample.SlideID}:")
        for filter in filters:
          if not filter:
            logger.info(filter.message)
    return result

  def processsample(self, sample, *, cleanup=False, **kwargs):
    with sample:
      if cleanup:
        sample.cleanup()
      return self.runsample(sample, **kwargs)

  def dryrun(self, **kwargs):
    """
    Print which samples would be run if you run the cohort
    """
    for samp, filters in self.sampledefswithfilters():
      logger = self.printlogger(samp)
      if all(filters):
        logger.info(f"{samp} would run")
      else:
        logger.info(f"{samp} would not run:")
        for filter in filters:
          if not filter:
            logger.info(filter.message)

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
      kwargs["slideidfilters"].append(SampleFilter(lambda self, sample, **kwargs: regex.match(sample.SlideID), f"SlideID matches {regex.pattern}", f"SlideID doesn't match {regex.pattern}"))
    return kwargs

class Im3Cohort(Cohort, Im3ArgumentParser):
  """
  Base class for any cohort that uses im3 files
  shardedim3root: the location of the sharded im3s
  """
  def __init__(self, root, shardedim3root, *args, **kwargs):
    super().__init__(root=root, *args, **kwargs)
    self.shardedim3root = pathlib.Path(shardedim3root)

  @property
  def root1(self): return self.root

  @property
  def rootnames(self):
    return {*super().rootnames, "shardedim3root"}

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "shardedim3root": self.shardedim3root}

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
    p.add_argument("--require-commit", type=thisrepo.getcommit, help="rerun samples that already finished with an AstroPath pipeline version earlier than this commit")
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
    require_commit = parsed_args_dict.pop("require_commit")

    if require_commit is not None and not require_commit.isancestor(thisrepo.currentcommit):
      raise ValueError(f"Trying to require commit {require_commit}, but that is not an ancestor of the current commit {thisrepo.currentcommit}")

    def filter(runstatus, dependencyrunstatuses):
      if skip_finished:
        cleanup = False
        if rerun_errors and runstatus.error is not None and not any(errorregex.search(runstatus.error) for errorregex in rerun_errors):
          runstatus.error = None
        if runstatus.started and require_commit is not None:
          if runstatus.gitcommit is None:
            raise ValueError("previous runstatus has gitcommit of None, check the log")
          if not require_commit <= runstatus.lastcleanstart:
            runstatus.started = runstatus.ended = False
        if not runstatus.started:  #log doesn't exist at all
          cleanup = True
        if runstatus.failedincleanup:
          cleanup = True
      else:
        #if we're rerunning, then we also want to clean up partially run files
        cleanup = True

      if not skip_finished and not dependencies:
        return FilterResult(True, "this filter is not run", cleanup=cleanup)

      elif skip_finished and not dependencies:
        if not runstatus:
          return FilterResult(True, "sample did not already run", cleanup=cleanup)
        else:
          return FilterResult(False, "sample already ran")

      elif dependencies and not skip_finished:
        for dependencyrunstatus in dependencyrunstatuses:
          if not dependencyrunstatus: return FilterResult(False, f"dependency {dependencyrunstatus.module} for {dependencyrunstatus.SlideID} "+str(dependencyrunstatus).replace('\n', ' '))
        return FilterResult(True, "all dependencies already ran", cleanup=cleanup)

      elif dependencies and skip_finished:
        for dependencyrunstatus in dependencyrunstatuses:
          if not dependencyrunstatus: return FilterResult(False, f"dependency {dependencyrunstatus.module} for {dependencyrunstatus.SlideID} "+str(dependencyrunstatus).replace('\n', ' '))
          if runstatus and not runstatus > dependencyrunstatus:
            runstatus.started = runstatus.ended = False #it's as if this step hasn't run
            cleanup = True
        if not runstatus:
          return FilterResult(True, "all dependencies already ran, sample has not run since then", cleanup=cleanup)
        else:
          return FilterResult(False, "sample already ran")
      else:
        assert False

    def slideidfilter(self, sample, **kwargs):
      return filter(
        runstatus=self.sampleclass.getrunstatus(SlideID=sample.SlideID, **self.workflowkwargs, **kwargs),
        dependencyrunstatuses=[
          dependency.getrunstatus(SlideID=sample.SlideID, **self.workflowkwargs)
          for dependency in self.sampleclass.workflowdependencyclasses()
        ],
      )
    kwargs["slideidfilters"].append(SampleFilter(slideidfilter, None, None))

    def samplefilter(self, sample, **kwargs):
      return filter(
        runstatus=sample.runstatus(),
        dependencyrunstatuses=[
          dependency.getrunstatus(SlideID=SlideID, **self.workflowkwargs, **kwargs)
          for dependency, SlideID in sample.workflowdependencies()
        ],
      )
    kwargs["samplefilters"].append(SampleFilter(samplefilter, None, None))

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
      logger = self.printlogger(sample)
      logger.info(f"{sample.SlideID} " + str(status).replace("\n", " "))
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
          result = super().processsample(sample, **kwargs)

          status = sample.runstatus(**kwargs)
          #we don't want to do anything if there's an error, because that
          #was already logged so no need to log it again and confuse the issue.
          if status.missingfiles and status.error is None:
            status.ended = True #to get the missing files in the __str__
            raise RuntimeError(f"{sample.logger.SlideID} {status}")

          return result

  @contextlib.contextmanager
  def handlesampledeffiltererror(self, samp, *, print_errors, **kwargs):
    if print_errors:
      try:
        yield
      except Exception as e:
        self.printlogger(samp).info(f"{samp.SlideID} gave an error in a sampledef filter: "+str(e).replace("\n", " "))
    else:
      with super().handlesampledeffiltererror(samp, **kwargs):
        yield

  @contextlib.contextmanager
  def handlesampleiniterror(self, samp, *, print_errors, **kwargs):
    if print_errors:
      try:
        yield
      except Exception as e:
        self.printlogger(samp).info(f"{samp.SlideID} gave an error in __init__: "+str(e).replace("\n", " "))
    else:
      with super().handlesampleiniterror(samp, **kwargs):
        yield

  @contextlib.contextmanager
  def handlesamplefiltererror(self, samp, *, print_errors, **kwargs):
    if print_errors:
      try:
        yield
      except Exception as e:
        self.printlogger(samp).info(f"{samp.SlideID} gave an error in a sample filter: "+str(e).replace("\n", " "))
    else:
      with super().handlesamplefiltererror(samp, **kwargs):
        yield

