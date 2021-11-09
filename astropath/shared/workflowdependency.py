import abc, contextlib, csv, datetime, more_itertools, re
from ..utilities.miscfileio import field_size_limit_context
from .logging import MyLogger

class ThingWithRoots(abc.ABC):
  @property
  def rootnames(self):
    return set()
  @property
  def rootkwargs(self):
    return {name: getattr(self, name) for name in self.rootnames}

class WorkflowDependency(ThingWithRoots):
  @property
  def workflowkwargs(self):
    return self.rootkwargs

  @classmethod
  @abc.abstractmethod
  def getoutputfiles(cls, **workflowkwargs):
    """
    Output files that this step is supposed to produce
    """
    return []

  @classmethod
  def getmissingoutputfiles(cls, **workflowkwargs):
    """
    Output files that were supposed to be produced but are missing
    """
    return [_ for _ in cls.getoutputfiles(**workflowkwargs) if not _.exists()]

  @property
  def outputfiles(self):
    """
    Output files that this step is supposed to produce
    """
    return self.getoutputfiles(**self.workflowkwargs)

  @property
  def missingoutputfiles(self):
    """
    Output files that were supposed to be produced but are missing
    """
    return self.getmissingoutputfiles(**self.workflowkwargs)

  @classmethod
  @abc.abstractmethod
  def getlogfile(cls, *, logroot, **workflowkwargs):
    pass
  @classmethod
  @abc.abstractmethod
  def usegloballogger(cls): pass

  @classmethod
  @abc.abstractmethod
  def logmodule(cls):
    "name of the log files for this class (e.g. align)"
  @classmethod
  @abc.abstractmethod
  def logstartregex(cls): pass
  @classmethod
  @abc.abstractmethod
  def logendregex(cls): pass

  @classmethod
  def getrunstatus(cls, *, SlideID, **workflowkwargs):
    workflowkwargs["SlideID"] = SlideID
    return SampleRunStatus.fromlog(SlideID=SlideID, samplelog=cls.getlogfile(**workflowkwargs), module=cls.logmodule(), missingfiles=cls.getmissingoutputfiles(**workflowkwargs), startregex=cls.logstartregex(), endregex=cls.logendregex())

  def runstatus(self, **kwargs):
    """
    returns a SampleRunStatus object that indicates whether
    the sample ran successfully or not, and information about
    the failure, if any.
    """
    return self.getrunstatus(**self.workflowkwargs, **kwargs)

  @property
  def rootnames(self):
    return {"logroot", *super().rootnames}

  @property
  @abc.abstractmethod
  def logroot(self):
    pass

  @abc.abstractmethod
  def run(self):
    pass

  @property
  @abc.abstractmethod
  def logger(self):
    pass
  @abc.abstractmethod
  def joblock(self):
    pass

  @abc.abstractmethod
  def workflowdependencies(self):
    return []

class WorkflowDependencySlideID(WorkflowDependency):
  @property
  def workflowkwargs(self):
    return {
      **super().workflowkwargs,
      "SlideID": self.SlideID,
    }

  @property
  @abc.abstractmethod
  def SlideID(self): pass

  @classmethod
  def getlogfile(cls, *, SlideID, logroot, **otherworkflowkwargs):
    return logroot/SlideID/"logfiles"/f"{SlideID}-{cls.logmodule()}.log"

class ExternalDependency(WorkflowDependencySlideID):
  def __init__(self, SlideID, logroot):
    self.__SlideID = SlideID
    self.__logroot = logroot
  @property
  def SlideID(self): return self.__SlideID
  @property
  def logroot(self): return self.__logroot
  @classmethod
  def getoutputfiles(cls, SlideID, **workflowkwargs): return super().getoutputfiles(SlideID, **workflowkwargs)

def makeexternaldependency(name, startregex, endregex):
  """
  for dependencies that don't run through this package
  """
  class dependency(ExternalDependency):
    @classmethod
    def logmodule(cls): return name
    @classmethod
    def logstartregex(cls): return startregex
    @classmethod
    def logendregex(cls): return endregex
  dependency.__name__ = name
  return dependency

ShredXML = makeexternaldependency("ShredXML", "shredxml started", "shredxml finished")

class SampleRunStatus:
  """
  Stores information about if a sample ran successfully.
  started: did it start running?
  ended: did it finish running?
  error: error traceback as a string, if any
  previousrun: SampleRunStatus for the previous run of this sample, if any
  missingfiles: files that are supposed to be in the output, but are missing
  """
  def __init__(self, *, module, started, ended, error=None, previousrun=None, missingfiles=()):
    self.module = module
    self.started = started
    self.ended = ended
    self.error = error
    self.previousrun = previousrun
    self.missingfiles = missingfiles
  def __bool__(self):
    """
    True if the sample started and ended with no error and all output files are present
    """
    return bool(self.started and self.ended and self.error is None and not self.missingfiles)
  @property
  def nruns(self):
    """
    How many times has this sample been run?
    """
    if self.previousrun is None:
      return 1 if self.started else 0
    return self.previousrun.nruns + 1

  @classmethod
  def fromlog(cls, *, samplelog, SlideID, module, missingfiles, startregex, endregex):
    """
    Create a SampleRunStatus object by reading the log file.
    samplelog: from CohortFolder/SlideID/logfiles/SlideID-module.log
               (not CohortFolder/logfiles/module.log)
    module: the module being run
    """
    result = None
    started = None
    ended = None
    previousrun = None
    error = None
    with contextlib.ExitStack() as stack:
      stack.enter_context(field_size_limit_context(1000000))
      try:
        f = stack.enter_context(open(samplelog))
      except IOError:
        return cls(started=None, ended=None, missingfiles=missingfiles, module=module)
      else:
        reader = more_itertools.peekable(csv.DictReader(f, fieldnames=("Project", "Cohort", "SlideID", "message", "time"), delimiter=";"))
        for row in reader:
          if row["SlideID"] != SlideID:
            continue
          elif not row["message"]:
            continue
          elif re.match(startregex, row["message"]):
            started = datetime.datetime.strptime(row["time"], MyLogger.dateformat)
            error = None
            ended = None
            previousrun = result
            result = None
          elif row["message"].startswith("ERROR:"):
            error = reader.peek(default={"message": ""})["message"]
            if error and error[0] == "[" and error[-1] == "]":
              error = "".join(eval(error))
            else:
              error = row["message"]
          elif re.match(endregex, row["message"]):
            ended = datetime.datetime.strptime(row["time"], MyLogger.dateformat)
            result = cls(started=started, ended=ended, error=error, previousrun=previousrun, missingfiles=missingfiles, module=module)
    if result is None:
      result = cls(started=started, ended=ended, error=error, previousrun=previousrun, missingfiles=missingfiles, module=module)
    return result

  def __str__(self):
    if self: return "ran successfully"
    if not self.started:
      return "did not run"
    elif self.error is not None:
      return "gave an error:\n\n"+self.error
    elif not self.ended:
      return "started, but did not end"
    elif self.missingfiles:
      return "ran successfully but some output files are missing: " + ", ".join(str(_) for _ in self.missingfiles)
    assert False, self
