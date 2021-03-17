import abc, contextlib, csv, more_itertools, re

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
  def getoutputfiles(cls, SlideID, **workflowkwargs):
    """
    Output files that this step is supposed to produce
    """
    return []

  @classmethod
  def getmissingoutputfiles(cls, SlideID, **workflowkwargs):
    """
    Output files that were supposed to be produced but are missing
    """
    return [_ for _ in cls.getoutputfiles(SlideID, **workflowkwargs) if not _.exists()]

  @property
  def outputfiles(self):
    """
    Output files that this step is supposed to produce
    """
    return self.getoutputfiles(self.SlideID, **self.workflowkwargs)

  @property
  def missingoutputfiles(self):
    """
    Output files that were supposed to be produced but are missing
    """
    return self.getmissingoutputfiles(self.SlideID, **self.workflowkwargs)

  @classmethod
  @abc.abstractmethod
  def logmodule(cls): pass


  @property
  @abc.abstractmethod
  def SlideID(self): pass

  @classmethod
  def getsamplelog(cls, SlideID, *, logroot, **otherworkflowkwargs):
    return logroot/SlideID/"logfiles"/f"{SlideID}-{cls.logmodule()}.log"

  @classmethod
  def getrunstatus(cls, SlideID, **workflowkwargs):
    return SampleRunStatus.fromlog(samplelog=cls.getsamplelog(SlideID, **workflowkwargs), module=cls.logmodule(), missingfiles=cls.getmissingoutputfiles(SlideID, **workflowkwargs), startregex=cls.logstartregex(), endregex=cls.logendregex())

  @classmethod
  @abc.abstractmethod
  def logstartregex(cls): pass
  @classmethod
  @abc.abstractmethod
  def logendregex(cls): pass

  @property
  def runstatus(self):
    """
    returns a SampleRunStatus object that indicates whether
    the sample ran successfully or not, and information about
    the failure, if any.
    """
    return self.getrunstatus(self.SlideID, **self.workflowkwargs)

  @property
  def rootnames(self):
    return {"logroot", *super().rootnames}

  @property
  @abc.abstractmethod
  def logroot(self):
    pass

class ExternalDependency(WorkflowDependency):
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
  def __init__(self, started, ended, error=None, previousrun=None, missingfiles=()):
    self.started = started
    self.ended = ended
    self.error = error
    self.previousrun = previousrun
    self.missingfiles = missingfiles
  def __bool__(self):
    """
    True if the sample started and ended with no error and all output files are present
    """
    return self.started and self.ended and self.error is None and not self.missingfiles
  @property
  def nruns(self):
    """
    How many times has this sample been run?
    """
    if self.previousrun is None:
      return 1 if self.started else 0
    return self.previousrun.nruns + 1

  @classmethod
  def fromlog(cls, *, samplelog, module, missingfiles, startregex, endregex):
    """
    Create a SampleRunStatus object by reading the log file.
    samplelog: from CohortFolder/SlideID/logfiles/SlideID-module.log
               (not CohortFolder/logfiles/module.log)
    module: the module being run
    """
    result = None
    started = False
    ended = False
    previousrun = None
    with contextlib.ExitStack() as stack:
      try:
        f = stack.enter_context(open(samplelog))
      except IOError:
        return cls(started=False, ended=False, missingfiles=missingfiles)
      else:
        reader = more_itertools.peekable(csv.DictReader(f, fieldnames=("Project", "Cohort", "SlideID", "message", "time"), delimiter=";"))
        for row in reader:
          if re.match(startregex, row["message"]):
            started = True
            error = None
            ended = False
            previousrun = result
            result = None
          elif row["message"].startswith("ERROR:"):
            error = reader.peek(default={"message": ""})["message"]
            if error and error[0] == "[" and error[-1] == "]":
              error = "".join(eval(error))
            else:
              error = row["message"]
          elif re.match(endregex, row["message"]):
            ended = True
            result = cls(started=started, ended=ended, error=error, previousrun=previousrun, missingfiles=missingfiles)
    if result is None:
      result = cls(started=started, ended=ended, error=error, previousrun=previousrun, missingfiles=missingfiles)
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
