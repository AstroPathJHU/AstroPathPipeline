import collections, functools, job_lock, logging, os, pathlib, traceback

class MyLogger:
  r"""
  Logger that follows the astropath logging conventions.
  It should always be used in a with statement that contains everything
  you want to do in the module.

  module: name of the process that's being run, e.g. "align"
  root: the Clinical_Specimen_* folder (or another folder if you want to log somewhere else)
  samp: the SampleDef object
  uselogfiles: should we write to log files (default: false)
  threshold: minimum level of messages that should be logged (default: logging.DEBUG)
  isglobal: is this a global log for the cohort? in that case samplelog is set to None and all the info goes in the mainlog
  mainlog: main log file, which gets errors and the most important warnings
           (default: root/logfiles/module.log)
  samplelog: sample log file, which gets errors, warnings, and info
             (default: root/SlideID/logfiles/SlideID-module.log)
  imagelog: image log file, which gets errors, warnings, and more info
            (default: None)
  reraiseexceptions: should the logger reraise exceptions after logging them
                     (default: True)

  To use the logger, you should write:
    with getlogger(module="module", root=r"\\bki0X\Clinical_Specimen_Y", samp="APIDXXXXXXXX", uselogfiles=True) as logger:
      #as soon as you enter the with statement, it will log "module v<version>" to both logs
      logger.critical("this gets logged to both logs")
      logger.error("this gets logged to both logs with ERROR: in front")
      #also if there is a python exception, it gets logged to both logs with ERROR: in front,
      # and the full traceback is logged to the sample log
      logger.warningglobal("this gets logged to both logs with WARNING: in front")
      logger.warning("this gets logged to the sample log with WARNING: in front")
      logger.info("this gets logged to the sample log")
      logger.imageinfo("this gets logged only to the image log, if it exists")
      logger.debug("this gets printed but doesn't go in any log")
      #when you exit the with statement, successfully or due to an error, it will log "end module"
  """
  def __init__(self, module, root, samp, *, uselogfiles=False, threshold=logging.DEBUG, isglobal=False, mainlog=None, samplelog=None, imagelog=None, reraiseexceptions=True):
    self.module = module
    self.root = pathlib.Path(root)
    self.samp = samp
    self.uselogfiles = uselogfiles
    self.nentered = 0
    self.threshold = threshold
    if mainlog is None:
      mainlog = self.root/"logfiles"/f"{self.module}.log"
    if samplelog is None:
      self.root
      self.samp.SlideID
      self.module
      samplelog = self.root/self.samp.SlideID/"logfiles"/f"{self.samp.SlideID}-{self.module}.log"
    self.mainlog = pathlib.Path(mainlog)
    self.samplelog = pathlib.Path(samplelog)
    self.imagelog = None if imagelog is None else pathlib.Path(imagelog)
    self.isglobal = isglobal
    self.reraiseexceptions = reraiseexceptions
    if uselogfiles and (self.Project is None or self.Cohort is None):
      raise ValueError("Have to give a non-None Project and Cohort when writing to log files")

    if not uselogfiles:
      self.__enter__()

  @property
  def SampleID(self): return self.samp.SampleID
  @property
  def SlideID(self): return self.samp.SlideID
  @property
  def Project(self): return self.samp.Project
  @property
  def Cohort(self): return self.samp.Cohort

  @property
  def formatter(self):
    return logging.Formatter(
      ";".join(str(_) for _ in (self.Project, self.Cohort, self.SlideID, "%(message)s", "%(asctime)s") if _ is not None),
      "%Y-%m-%d %H:%M:%S",
    )
  def __enter__(self):
    if self.nentered == 0:
      self.logger = logging.getLogger(f"{self.root}.{self.module}.{self.Project}.{self.Cohort}.{self.SlideID}.{self.uselogfiles}.{self.threshold}.{self.isglobal}")
      self.logger.setLevel(self.threshold)

      printhandler = logging.StreamHandler()
      printhandler.setFormatter(self.formatter)
      printhandler.addFilter(self.filter)
      printhandler.setLevel(logging.DEBUG)
      self.logger.addHandler(printhandler)

      if self.uselogfiles:
        self.mainlog.parent.mkdir(exist_ok=True, parents=True)
        mainhandler = MyFileHandler(self.mainlog)
        mainhandler.setFormatter(self.formatter)
        mainhandler.addFilter(self.filter)
        mainhandler.setLevel(logging.INFO if self.isglobal else logging.WARNING+1)
        self.logger.addHandler(mainhandler)

        if not self.isglobal:
          self.samplelog.parent.mkdir(exist_ok=True, parents=True)
          samplehandler = MyFileHandler(self.samplelog)
          samplehandler.setFormatter(self.formatter)
          samplehandler.addFilter(self.filter)
          samplehandler.setLevel(logging.INFO)
          self.logger.addHandler(samplehandler)

        if self.imagelog is not None :
          self.imagelog.parent.mkdir(exist_ok=True, parents=True)
          imagehandler = MyFileHandler(self.imagelog)
          imagehandler.setFormatter(self.formatter)
          imagehandler.addFilter(self.filter)
          imagehandler.setLevel(logging.INFO-1)
          self.logger.addHandler(imagehandler)

        from ..utilities.version import astropathversion
        self.logger.critical(f"{self.module} {astropathversion}")

    self.nentered += 1
    return self

  def filter(self, record):
    try:
      levelname = {
        logging.WARNING: "WARNING",
        logging.WARNING+1: "WARNING",
        logging.ERROR: "ERROR",
      }[record.levelno]
    except KeyError:
      pass
    else:
      if not record.msg.startswith(levelname+": "):
        record.msg = f"{levelname}: {record.msg}"
    if ";" in record.msg or "\n" in record.msg:
      raise ValueError("log messages aren't supposed to have semicolons or newlines:\n\n"+record.msg)
    return True

  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.nentered -= 1
    if self.nentered == 0:
      if exc_value is not None:
        errormessage = str(exc_value).replace(";", ",")
        if "\n" in errormessage: errormessage = repr(errormessage)
        self.error(errormessage)
        self.info(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)).replace(";", ""))
      self.logger.info(f"end {self.module}")
      for handler in self.handlers[:]:
        handler.close()
        self.removeHandler(handler)
      del self.logger
    return self.nentered == 0 and not self.reraiseexceptions

  def __getattr__(self, attr):
    if attr == "logger":
      raise RuntimeError("Have to use this in a context manager if you want to uselogfiles")
    return getattr(self.logger, attr)
  def warningglobal(self, *args, **kwargs):
    """
    A warning that also goes in the main log
    """
    return self.logger.log(logging.WARNING+1, *args, **kwargs)
  def imageinfo(self, *args, **kwargs):
    """
    An info message that only goes in the image log
    """
    return self.logger.log(logging.INFO-1, *args, **kwargs)

class MyFileHandler:
  """
  Allows the same file to be used by multiple loggers
  without conflicting with each other
  """
  __handlers = {}
  __counts = collections.Counter()

  def __init__(self, filename):
    self.__filename = pathlib.Path(filename)
    self.__lockfilename = self.__filename.with_suffix(self.__filename.suffix+".lock")
    if filename not in self.__handlers:
      handler = self.__handlers[filename] = logging.FileHandler(filename, delay=True)
      kwargs = {"newline": "", "mode": handler.mode, "encoding": handler.encoding}
      try:
        kwargs["errors"] = handler.errors
      except AttributeError: #python < 3.9
        pass
      newfile = open(filename, **kwargs)
      try:
        handler.setStream(newfile)
      except AttributeError: #python < 3.7
        handler.stream = newfile
      handler.terminator = "\r\n"
    self.__handler = self.__handlers[filename]
    self.__counts[filename] += 1
    self.__formatter = self.__handler.formatter
    self.__filters = self.__handler.filters
    self.__level = self.__handler.level

  def close(self):
    self.__counts[self.__filename] -= 1
    if not self.__counts[self.__filename]:
      self.__handler.close()
      del self.__handlers[self.__filename]

  def setFormatter(self, formatter):
    self.__formatter = formatter
  def addFilter(self, filter):
    if filter not in self.__filters:
      self.__filters.append(filter)
  def setLevel(self, level):
    self.__level = level

  @property
  def level(self):
    return self.__level

  def handle(self, record):
    if os.fspath(self.__filename) == os.fspath(os.devnull): return
    self.__handler.setFormatter(self.__formatter)
    self.__handler.setLevel(self.__level)
    self.__handler.filters = self.__filters
    with job_lock.JobLockAndWait(self.__lockfilename, 1, task=f"logging to {self.__filename}"):
      self.__handler.handle(record)

  def __repr__(self):
    return f"{type(self).__name__}({self.__filename})"

__notgiven = object()

@functools.lru_cache(maxsize=None)
def __getlogger(*, module, root, samp, uselogfiles, threshold, isglobal, mainlog, samplelog, imagelog, reraiseexceptions):
  return MyLogger(module, root, samp, uselogfiles=uselogfiles, threshold=threshold, isglobal=isglobal, mainlog=mainlog, samplelog=samplelog, imagelog=imagelog, reraiseexceptions=reraiseexceptions)

def getlogger(*, module, root, samp, uselogfiles=False, threshold=logging.DEBUG, isglobal=False, mainlog=None, samplelog=None, imagelog=None, reraiseexceptions=True):
  from .sample import SampleDef
  samp = SampleDef(root=root, samp=samp)
  return __getlogger(module=module, root=root, samp=samp, uselogfiles=uselogfiles, threshold=threshold, isglobal=isglobal, mainlog=mainlog, samplelog=samplelog, imagelog=imagelog, reraiseexceptions=reraiseexceptions)
