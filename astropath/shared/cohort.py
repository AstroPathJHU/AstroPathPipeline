import abc, contextlib, datetime, job_lock, logging, pathlib, re
try:
  contextlib.nullcontext
except AttributeError:
  import contextlib2 as contextlib
from ..utilities.config import CONST as UNIV_CONST
from ..utilities import units
from ..utilities.tableio import readtable, writetable
from ..utilities.version.git import thisrepo
from .argumentparser import ArgumentParserMoreRoots, DbloadArgumentParser, DeepZoomArgumentParser, GeomFolderArgumentParser, Im3ArgumentParser, ImageCorrectionArgumentParser, MaskArgumentParser, ParallelArgumentParser, RunFromArgumentParser, SegmentationFolderArgumentParser, SelectLayersArgumentParser, SelectRectanglesArgumentParser, TempDirArgumentParser, XMLPolygonFileArgumentParser, ZoomFolderArgumentParser
from .astropath_logging import getlogger, ThingWithLogger
from .rectangle import rectanglefilter
from .workflowdependency import ThingWithRoots, ThingWithWorkflowKwargs, WorkflowDependency

class CohortBase(ThingWithRoots, ThingWithLogger):
  """
  Base class for a cohort.  This class doesn't actually run anything
  (for that use Cohort, below).
  """
  def __init__(self, *args, root, sampledefroot=None, logroot=None, uselogfiles=True, reraiseexceptions=False, moremainlogroots=[], skipstartfinish=False, printthreshold=logging.NOTSET-100, runfromapid=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.__root = pathlib.Path(root)
    if logroot is None: logroot = self.__root
    self.__logroot = pathlib.Path(logroot)
    if sampledefroot is None: sampledefroot = self.__root
    self.__sampledefroot = pathlib.Path(sampledefroot)
    self.__uselogfiles = uselogfiles
    self.reraiseexceptions = reraiseexceptions
    self.moremainlogroots = moremainlogroots
    self.skipstartfinish = skipstartfinish
    self.printthreshold = printthreshold
    self.runfromapid = runfromapid

  def tissuesampledefs(self, **kwargs):
    from .samplemetadata import APIDDef, SampleDef
    if self.runfromapid:
      csvfiles = (self.sampledefroot/"upkeep_and_progress").glob("AstropathAPIDdef_*.csv")
      csvclass = APIDDef
    else:
      csvfiles = [self.sampledefroot/"sampledef.csv"]
      csvclass = SampleDef
    for csvfile in csvfiles:
      yield from readtable(csvfile, csvclass)
  def controlsampledefs(self, **kwargs):
    from .samplemetadata import ControlTMASampleDef
    csvfiles = (self.sampledefroot/"Ctrl").glob("[Pp]roject*_ctrlsamples.csv")
    csvclass = ControlTMASampleDef
    for csvfile in csvfiles:
      yield from readtable(csvfile, csvclass)

  def sampledefs(self, **kwargs):
    if self.usetissue:
      yield from self.tissuesampledefs(**kwargs)
    if self.useTMA:
      yield from self.controlsampledefs(**kwargs)

  @property
  @abc.abstractmethod
  def usetissue(self): pass
  @property
  @abc.abstractmethod
  def useTMA(self): pass

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

  def globallogger(self, *, SlideID=None, **kwargs):
    from .samplemetadata import SampleDef
    if SlideID is None: SlideID = f"project{self.Project}"
    samp = SampleDef(Project=self.Project, Cohort=self.Cohort, SlideID=SlideID)
    return self.getlogger(samp, isglobal=True, **kwargs)

  def printlogger(self, *args, **kwargs): return self.getlogger(*args, uselogfiles=False, skipstartfinish=True, **kwargs)

  @property
  def logger(self): return self.globallogger()
  @property
  def globalprintlogger(self): return self.globallogger(uselogfiles=False)
  @property
  def mainlogs(self): return self.logger.mainlogs

  def globaljoblock(self, corruptfiletimeout=datetime.timedelta(minutes=10), **kwargs):
    lockfiles = [logfile.with_suffix(".lock") for logfile in self.globallogger().mainlogs]
    return job_lock.MultiJobLock(*lockfiles, corruptfiletimeout=corruptfiletimeout, mkdir=True, **kwargs)

  def getlogger(self, samp, *, isglobal=False, uselogfiles=True, skipstartfinish=False, **kwargs):
    if isinstance(samp, WorkflowDependency):
      isglobal = isglobal or samp.usegloballogger()
      samp = samp.samp
    return getlogger(module=self.logmodule(), root=self.logroot, samp=samp, uselogfiles=uselogfiles and self.__uselogfiles, reraiseexceptions=self.reraiseexceptions, isglobal=isglobal, moremainlogroots=self.moremainlogroots, skipstartfinish=skipstartfinish or self.skipstartfinish, printthreshold=self.printthreshold, sampledefroot=self.sampledefroot, **kwargs)

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
    p.add_argument("--sampledefroot", "--apiddefroot", type=pathlib.Path, help="folder to look for sampledef.csv and/or the upkeep_and_progress folder")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-apiddef", action="store_true", dest="runfromapid", help="use AstropathAPIDdef_Project.csv to get the list of samples to run")
    g.add_argument("--use-sampledef", action="store_false", dest="runfromapid", help="use sampledef.csv to get the list of samples to run")
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
      "runfromapid": dct.pop("runfromapid"),
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

class Cohort(RunCohortBase, ArgumentParserMoreRoots, ThingWithWorkflowKwargs, contextlib.ExitStack):
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
  def __init__(self, *args, slideidfilters=[], samplefilters=[], im3root=None, debug=False, informdataroot=None, batchroot=None, xmlfolders=[], **kwargs):
    super().__init__(*args, reraiseexceptions=debug, **kwargs)
    if im3root is None: im3root = self.root
    self.im3root = pathlib.Path(im3root)
    if informdataroot is None: informdataroot = self.root
    self.informdataroot = pathlib.Path(informdataroot)
    if batchroot is None: batchroot = self.root
    self.batchroot = pathlib.Path(batchroot)
    self.slideidfilters = slideidfilters
    self.samplefilters = samplefilters
    self.xmlfolders = xmlfolders

  def sampledefswithfilters(self, *, check_all_filters=False, **kwargs):
    for samp in self.sampledefs():
      filterresults = []
      exceptions = []
      for filter in self.slideidfilters:
        try:
          filterresults.append(filter(self, samp, **kwargs))
        except Exception as e:
          exceptions.append(e)
        if not check_all_filters and not all(filterresults):
          break
      if exceptions and all(filterresults):
        #with self.handlesampledeffiltererror(samp, **kwargs):
          raise exceptions[0]
      yield samp, filterresults

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

  def allsamples(self, *, moreinitkwargs={}, **kwargs) :
    if moreinitkwargs is None: moreinitkwargs = {}
    for samp in self.sampledefs(**kwargs):
      try:
        sample = self.initiatesample(samp, **moreinitkwargs)
        if sample.logmodule() != self.logmodule():
          raise ValueError(f"Wrong logmodule: {self.logmodule()} != {sample.logmodule()}")
        yield sample
      except Exception:
        #enter the logger here to log exceptions in __init__ of the sample
        #but not KeyboardInterrupt
        with self.handlesampleiniterror(samp, **kwargs):
          raise

  def samplesandsampledefswithfilters(self, *, check_all_filters=False, moreinitkwargs=None, **kwargs):
    if moreinitkwargs is None: moreinitkwargs = {}
    for samp, filters in self.sampledefswithfilters(check_all_filters=check_all_filters, **kwargs):
      if all(filters):
        try:
          sample = self.initiatesample(samp, **moreinitkwargs)
          if sample.logmodule() != self.logmodule():
            raise ValueError(f"Wrong logmodule: {self.logmodule()} != {sample.logmodule()}")
        except Exception:
          #enter the logger here to log exceptions in __init__ of the sample
          #but not KeyboardInterrupt
          with self.handlesampleiniterror(samp, **kwargs):
            raise
        else:
          for filter in self.samplefilters:
            try:
              filters.append(filter(self, sample, **kwargs))
            except Exception:
              #enter the logger here to log exceptions in __init__ of the sample
              #but not KeyboardInterrupt
              with self.handlesamplefiltererror(samp, **kwargs):
                raise
            if not check_all_filters and not all(filters): break
          yield sample, filters
      else:
        yield samp, filters

  def sampleswithfilters(self, *, printnotrunning=False, **kwargs):
    for samp, filters in self.samplesandsampledefswithfilters(printnotrunning=printnotrunning, **kwargs):
      if isinstance(samp, WorkflowDependency):
        yield samp, filters
      elif printnotrunning:
        logger = self.printlogger(samp)
        logger.info(f"Not running {samp.SlideID}:")
        for filter in filters:
          if not filter:
            logger.info(filter.message)

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
  TMAsampleclass = None

  @classmethod
  def sampleclassforsampledef(cls, samp):
    from .samplemetadata import APIDDef, ControlTMASampleDef, SampleDef
    return {
      SampleDef: cls.sampleclass,
      APIDDef: cls.sampleclass,
      ControlTMASampleDef: cls.TMAsampleclass,
    }[type(samp)]

  @property
  def usetissue(self): return self.sampleclass is not None
  @property
  def useTMA(self): return self.TMAsampleclass is not None

  def initiatesample(self, samp, **morekwargs):
    "Create a Sample object (subclass of SampleBase) from SampleDef samp to run on"
    sampleclass = self.sampleclassforsampledef(samp)
    return sampleclass(samp=samp, **self.initiatesamplekwargs, **morekwargs)

  @property
  def initiatesamplekwargs(self):
    "Keyword arguments to pass to the sample class"
    return {
      "root": self.root,
      "reraiseexceptions": self.reraiseexceptions,
      "uselogfiles": self.uselogfiles,
      "logroot": self.logroot,
      "im3root": self.im3root,
      "informdataroot": self.informdataroot,
      "batchroot": self.batchroot,
      "xmlfolders": self.xmlfolders,
      "moremainlogroots": self.moremainlogroots,
      "skipstartfinish": self.skipstartfinish,
      "printthreshold": self.printthreshold,
      "sampledefroot": self.sampledefroot,
    }

  @classmethod
  def logmodule(cls):
    "name of the log files for this class (e.g. align)"
    result, = {
      sampleclass.logmodule()
      for sampleclass in (cls.sampleclass, cls.TMAsampleclass)
      if sampleclass is not None
    }
    return result

  @property
  def rootnames(self):
    return {*super().rootnames, "im3root", "informdataroot", "batchroot"}
  @property
  def workflowkwargs(self):
    return {
      "xmlfolder": None,
      **super().workflowkwargs,
      **self.rootkwargs,
    }

  def run(self, *, cleanup=False, printnotrunning=True, check_all_filters=False, moreinitkwargs=None, **kwargs):
    """
    Run the cohort by iterating over the samples and calling runsample on each.
    """
    result = []
    if moreinitkwargs is None: moreinitkwargs = {}
    for sample, filters in self.samplesandsampledefswithfilters(printnotrunning=printnotrunning, check_all_filters=check_all_filters, moreinitkwargs=moreinitkwargs, **kwargs):
      result.append(self.processsample(sample, filters=filters, cleanup=cleanup, printnotrunning=printnotrunning, **kwargs))
    return result

  def processsample(self, sample, *, filters, printnotrunning=True, cleanup=False, **kwargs):
    if all(filters):
      with sample:
        if cleanup or any(_.cleanup for _ in filters):
          sample.cleanup()
        return self.runsample(sample, **kwargs)
    elif printnotrunning:
      logger = self.printlogger(sample)
      logger.info(f"Not running {sample.SlideID}:")
      for filter in filters:
        if not filter:
          logger.info(filter.message)

  def dryrun(self, *, cleanup=False, **kwargs):
    """
    Print which samples would be run if you run the cohort
    """
    for samp, filters in self.sampledefswithfilters(**kwargs):
      logger = self.printlogger(samp)
      if all(filters):
        if any(filter.cleanup for filter in filters):
          logger.info(f"{samp} would cleanup and run")
        else:
          logger.info(f"{samp} would run")
      else:
        logger.info(f"{samp} would not run:")
        for filter in filters:
          if not filter:
            logger.info(filter.message)

  @classmethod
  def defaultunits(cls):
    result, = {
      sampleclass.defaultunits()
      for sampleclass in (cls.sampleclass, cls.TMAsampleclass)
      if sampleclass is not None
    }
    return result

  @classmethod
  def makeargumentparser(cls, **kwargs):
    """
    Create an argument parser to run this cohort on the command line
    """
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--debug", action="store_true", help="exit on errors, instead of logging them and continuing")
    p.add_argument("--include-bad-samples", action="store_true", help="include samples that have isGood set to False")
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
    if not dct.pop("include_bad_samples"):
      kwargs["slideidfilters"].append(SampleFilter(lambda self, sample, **kwargs: bool(sample), "Sample is good", "Sample is not good"))
    regex = dct.pop("sampleregex")
    if regex is not None:
      kwargs["slideidfilters"].append(SampleFilter(lambda self, sample, **kwargs: regex.match(sample.SlideID), f"SlideID matches {regex.pattern}", f"SlideID doesn't match {regex.pattern}"))
    return kwargs

class Im3Cohort(Cohort, Im3ArgumentParser):
  """
  Base class for any cohort that uses im3 files
  shardedim3root: the location of the sharded im3s
  im3filetype: file type of the im3s (raw or flatWarp)
  """
  def __init__(self, root, shardedim3root, *args, im3filetype, **kwargs):
    super().__init__(root=root, *args, **kwargs)
    self.shardedim3root = pathlib.Path(shardedim3root)
    self.im3filetype = im3filetype

  @property
  def root1(self): return self.root

  @property
  def rootnames(self):
    return {*super().rootnames, "shardedim3root"}

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "shardedim3root": self.shardedim3root,
      "im3filetype": self.im3filetype,
    }

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

class GlobalDbloadCohortBase(DbloadCohortBase):
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
  def __init__(self, *args, zoomroot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if zoomroot is None: zoomroot = self.root
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

class SelectLayersIm3Cohort(SelectLayersCohort):
  @property
  def workflowkwargs(self):
    result = {
      **super().workflowkwargs,
      "layersim3": self.layers,
    }
    if len(self.layers) == 1 and self.layers is not None:
      layer, = self.layers
      result["layerim3"] = layer
    return result

class SegmentationFolderCohort(Cohort, SegmentationFolderArgumentParser):
  def __init__(self, *args, segmentationfolder=None, segmentationroot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if segmentationfolder is not None: segmentationfolder = pathlib.Path(segmentationfolder)
    if segmentationroot is None: segmentationroot = self.im3root
    segmentationroot = pathlib.Path(segmentationroot)
    self.segmentationfolder = segmentationfolder
    self.segmentationroot = segmentationroot

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "segmentationroot": self.segmentationroot, "segmentationfolder": self.segmentationfolder}

  @property
  def workflowkwargs(self) :
    return {
      **super().workflowkwargs,
      'segmentationfolderarg':self.segmentationfolder,
      'segmentationroot':self.segmentationroot,
    }

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

class XMLPolygonFileCohort(Cohort, XMLPolygonFileArgumentParser):
  def __init__(self, *args, annotationsxmlregex=None, **kwargs):
    self.__annotationsxmlregex = annotationsxmlregex
    super().__init__(*args, **kwargs)
  @property
  def workflowkwargs(self):
    return {
      **super().workflowkwargs,
      "annotationsxmlregex": self.__annotationsxmlregex,
    }
  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "annotationsxmlregex": self.__annotationsxmlregex,
    }

class CorrectedImageCohort(Im3Cohort,ImageCorrectionArgumentParser) :
  """
  Class for a cohort that uses corrected im3 images as its sample rectangles
  """
  def __init__(self,*args,et_offset_file,skip_et_corrections,flatfield_file,warping_file,correction_model_file,**kwargs) :
    super().__init__(*args,**kwargs)
    self.__et_offset_file = et_offset_file
    self.__skip_et_corrections = skip_et_corrections
    self.__flatfield_file = flatfield_file
    self.__warping_file = warping_file
    self.__correction_model_file = correction_model_file
  @property
  def initiatesamplekwargs(self) :
    return {
      **super().initiatesamplekwargs,
      'et_offset_file': self.__et_offset_file,
      'skip_et_corrections':self.__skip_et_corrections,
      'flatfield_file': self.__flatfield_file,
      'warping_file': self.__warping_file,
      'correction_model_file': self.__correction_model_file,
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
    g = p.add_mutually_exclusive_group()
    g.add_argument("--print-errors", action="store_const", help="instead of running samples, print the status of the ones that haven't run, including error messages", dest="print_mode", const="errors")
    g.add_argument("--print-all", action="store_const", help="instead of running samples, print the status of all the samples, including error messages", dest="print_mode", const="all")
    g.add_argument("--print-timing", action="store_const", help="instead of running samples, print the status of the samples that finished already", dest="print_mode", const="timing")
    p.add_argument("--ignore-error", type=re.compile, action="append", dest="ignore_errors", help="for --print-errors, ignore any errors that match this regex")
    p.add_argument("--check-all-filters", action="store_true", help="check all filters, even if one of them fails, in order to print more info")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
    }
    if parsed_args_dict["print_mode"]:
      kwargs["uselogfiles"] = False
      parsed_args_dict["skip_finished"] = parsed_args_dict["dependencies"] = False

    dependencies = parsed_args_dict.pop("dependencies")
    skip_finished = parsed_args_dict.pop("skip_finished")
    rerun_errors = parsed_args_dict.pop("rerun_errors")
    require_commit = parsed_args_dict.pop("require_commit")

    if require_commit is not None and not require_commit.isancestor(thisrepo.currentcommit):
      raise ValueError(f"Trying to require commit {require_commit}, but that is not an ancestor of the current commit {thisrepo.currentcommit}")

    runstatusfilterkwargs = {
      "skip_finished": skip_finished,
      "dependencies": dependencies,
      "rerun_errors": rerun_errors,
      "require_commit": require_commit,
    }

    def slideidfilter(self, sample, **kwargs):
      sampleclass = self.sampleclassforsampledef(sample)
      return runstatusfilter(
        runstatus=sampleclass.getrunstatus(SlideID=sample.SlideID, Scan=sample.Scan, BatchID=sample.BatchID, **self.workflowkwargs, **kwargs),
        dependencyrunstatuses=[
          dependency.getrunstatus(SlideID=sample.SlideID, Scan=sample.Scan, BatchID=sample.BatchID, **self.workflowkwargs)
          for dependency in sampleclass.workflowdependencyclasses(SlideID=sample.SlideID, Scan=sample.Scan, **self.workflowkwargs)
        ],
        **runstatusfilterkwargs,
      )
    kwargs["slideidfilters"].append(SampleFilter(slideidfilter, None, None))

    def samplefilter(self, sample, **kwargs):
      return runstatusfilter(
        runstatus=sample.runstatus(),
        dependencyrunstatuses=[
          dependency.getrunstatus(SlideID=SlideID, Scan=sample.samp.Scan, BatchID=sample.samp.BatchID, **self.workflowkwargs, **kwargs)
          for dependency, SlideID in sample.workflowdependencies(SlideID=sample.SlideID, Scan=sample.samp.Scan, BatchID=sample.samp.BatchID, **self.workflowkwargs)
        ],
        **runstatusfilterkwargs,
      )
    kwargs["samplefilters"].append(SampleFilter(samplefilter, None, None))

    def inputfilesfilter(self, sample, **kwargs):
      missinginputs = [file for file in sample.inputfiles(**kwargs) if not file.exists()]
      if missinginputs:
        return FilterResult(False, "missing input files: " + ", ".join(str(_) for _ in missinginputs))
      return FilterResult(True, "all input files exist", cleanup=False)

    if not parsed_args_dict["print_mode"]:
      kwargs["samplefilters"].append(SampleFilter(inputfilesfilter, None, None))

    return kwargs

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "print_mode": parsed_args_dict.pop("print_mode"),
      "ignore_errors": parsed_args_dict.pop("ignore_errors"),
      "check_all_filters": parsed_args_dict.pop("check_all_filters"),
    }
    return kwargs

  def processsample(self, sample, *, filters, print_mode, ignore_errors, **kwargs):
    passedfilters = all(filters) and isinstance(sample, WorkflowDependency)

    if print_mode:
      if ignore_errors is None: ignore_errors = []

      logger = self.printlogger(sample)

      message = None
      for filter in filters:
        if not filter:
          if filter.message.startswith("sample already ran"):
            if print_mode == "errors": return
            elif print_mode == "all": pass
            elif print_mode == "timing": pass
            else: assert False, print_mode
          if filter.message == "Sample is not good": return
          message = f"{sample.SlideID} {filter.message}"
          break
      if message is not None:
        if print_mode == "timing" and not "sample already ran" in message:
          return
        logger.info(message)
        return

      if print_mode == "timing": return
      status = sample.runstatus(**kwargs)
      if status and print_mode == "errors": return
      if status.error and any(ignore.search(status.error) for ignore in ignore_errors): return
      logger.info(f"{sample.SlideID} " + str(status).replace("\n", " "))
    else:
      with sample.joblock() if passedfilters else contextlib.nullcontext(True) as lock:
        if not lock:
          logger = self.printlogger(sample)
          logger.info(f"Not running {sample.SlideID}:")
          logger.info(f"{sample.SlideID} is already being run by another process")
          return
        if passedfilters:
          try:
            missinginputs = [file for file in sample.inputfiles(**kwargs) if not file.exists()]
            if missinginputs:
              assert False #should not be able to get here anymore
              raise IOError("Not all required input files exist.  Missing files: " + ", ".join(str(_) for _ in missinginputs))
          except Exception: #don't log KeyboardInterrupt here
            with sample:
              raise
            return

        with self.getlogger(sample) if passedfilters else contextlib.nullcontext():
          result = super().processsample(sample, filters=filters, **kwargs)

          if passedfilters:
            status = sample.runstatus(**kwargs)
            #we don't want to do anything if there's an error, because that
            #was already logged so no need to log it again and confuse the issue.
            if status.missingfiles and status.error is None:
              status.started = status.ended = True #to get the missing files in the __str__
              raise RuntimeError(f"{sample.logger.SlideID} {status}")

          return result

  def run(self, *, print_mode=None, printnotrunning=None, moreinitkwargs=None, **kwargs):
    if moreinitkwargs is None: moreinitkwargs = {}
    moreinitkwargs = dict(moreinitkwargs)
    if print_mode:
      if printnotrunning is None:
        kwargs["printnotrunning"] = False
      moreinitkwargs["suppressinitwarnings"] = True
    return super().run(print_mode=print_mode, moreinitkwargs=moreinitkwargs, **kwargs)

  @contextlib.contextmanager
  def handlesampledeffiltererror(self, samp, *, print_mode, **kwargs):
    if print_mode:
      try:
        yield
      except Exception as e:
        self.printlogger(samp).info(f"{samp.SlideID} gave an error in a sampledef filter: "+str(e).replace("\n", " "))
    else:
      with super().handlesampledeffiltererror(samp, **kwargs):
        yield

  @contextlib.contextmanager
  def handlesampleiniterror(self, samp, *, print_mode=None, **kwargs):
    if print_mode:
      try:
        yield
      except Exception as e:
        self.printlogger(samp).info(f"{samp.SlideID} gave an error in __init__: "+str(e).replace("\n", " "))
    else:
      with super().handlesampleiniterror(samp, **kwargs):
        yield

  @contextlib.contextmanager
  def handlesamplefiltererror(self, samp, *, print_mode, **kwargs):
    if print_mode:
      try:
        yield
      except Exception as e:
        self.printlogger(samp).info(f"{samp.SlideID} gave an error in a sample filter: "+str(e).replace("\n", " "))
    else:
      with super().handlesamplefiltererror(samp, **kwargs):
        yield


def runstatusfilter(*, runstatus, dependencyrunstatuses, skip_finished, dependencies, rerun_errors, require_commit):
  if isinstance(runstatus, Exception):
    return FilterResult(False, f"runstatus gave an error: {runstatus}", cleanup=False)
  for dep in dependencyrunstatuses:
    if isinstance(dep, Exception):
      return FilterResult(False, f"dependency runstatus gave an error: {dep}", cleanup=False)

  if skip_finished:
    cleanup = False
    if rerun_errors and runstatus.error is not None and not any(errorregex.search(runstatus.error) for errorregex in rerun_errors):
      runstatus.error = None
    if runstatus.started and require_commit is not None:
      if runstatus.gitcommit is None:
        runstatus.started = runstatus.ended = False
      elif not require_commit <= runstatus.lastcleanstart:
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
      return FilterResult(False, f"sample already ran in {runstatus.ended - runstatus.started}")

  elif dependencies and not skip_finished:
    for dependencyrunstatus in dependencyrunstatuses:
      if not dependencyrunstatus: return FilterResult(False, f"dependency {dependencyrunstatus.module} for {dependencyrunstatus.SlideID} "+str(dependencyrunstatus).replace('\n', ' '))
      if not thisrepo.currentcommit >= dependencyrunstatus: return FilterResult(False, f"current commit {thisrepo.currentcommit} is not descended from {dependencyrunstatus.module} commit {dependencyrunstatus.gitcommit}")
    return FilterResult(True, "all dependencies already ran", cleanup=cleanup)

  elif dependencies and skip_finished:
    for dependencyrunstatus in dependencyrunstatuses:
      if not dependencyrunstatus: return FilterResult(False, f"dependency {dependencyrunstatus.module} for {dependencyrunstatus.SlideID} "+str(dependencyrunstatus).replace('\n', ' '))
      if not thisrepo.currentcommit >= dependencyrunstatus: return FilterResult(False, f"current commit {thisrepo.currentcommit} is not descended from {dependencyrunstatus.module} commit {dependencyrunstatus.gitcommit}")
      if runstatus.started and not runstatus.lastcleanstart > dependencyrunstatus:
        runstatus.started = runstatus.ended = False #it's as if this step hasn't run
        cleanup = True
    if not runstatus:
      return FilterResult(True, "all dependencies already ran, sample has not run since then", cleanup=cleanup)
    else:
      return FilterResult(False, f"sample already ran in {runstatus.ended - runstatus.started}")
  else:
    assert False

