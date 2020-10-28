import collections, functools, logging, pathlib

class MyLogger:
  def __init__(self, module, root, samp, *, uselogfiles=False, threshold=logging.DEBUG, mainlog=None, samplelog=None):
    self.module = module
    self.root = pathlib.Path(root)
    self.samp = samp
    self.uselogfiles = uselogfiles
    self.nentered = 0
    self.threshold = threshold
    if mainlog is None:
      mainlog = self.root/"logfiles"/f"{self.module}.log"
    if samplelog is None:
      samplelog = self.root/self.SlideID/"logfiles"/f"{self.SlideID}-{self.module}.log"
    self.mainlog = pathlib.Path(mainlog)
    self.samplelog = pathlib.Path(samplelog)
    if uselogfiles and (self.Project is None or self.SampleID is None or self.Cohort is None):
      raise ValueError("Have to give a non-None SampleID, Project, and Cohort when writing to log files")

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
      ";".join(str(_) for _ in (self.Project, self.Cohort, self.SampleID, self.SlideID, "%(message)s", "%(asctime)s") if _ is not None),
      "%Y-%b-%d %H:%M:%S",
    )
  def __enter__(self):
    if self.nentered == 0:
      self.logger = logging.getLogger(f"{self.root}.{self.module}.{self.Project}.{self.Cohort}.{self.SlideID}.{self.uselogfiles}.{self.threshold}")
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
        mainhandler.setLevel(logging.WARNING+1)
        self.logger.addHandler(mainhandler)

        self.samplelog.parent.mkdir(exist_ok=True, parents=True)
        samplehandler = MyFileHandler(self.samplelog)
        samplehandler.setFormatter(self.formatter)
        samplehandler.addFilter(self.filter)
        samplehandler.setLevel(logging.INFO)
        self.logger.addHandler(samplehandler)

        self.logger.critical(self.module)

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
    if ";" in record.msg:
      raise ValueError("log messages aren't supposed to have semicolons:\n\n"+record.msg)
    return True

  def __exit__(self, *exc):
    self.nentered -= 1
    if self.nentered == 0:
      self.logger.info(f"end {self.module}")
      for handler in self.handlers[:]:
        handler.close()
        self.removeHandler(handler)
      del self.logger

  def __getattr__(self, attr):
    if attr == "logger":
      raise RuntimeError("Have to use this in a context manager if you want to uselogfiles")
    return getattr(self.logger, attr)
  def warningglobal(self, *args, **kwargs):
    return self.logger.log(logging.WARNING+1, *args, **kwargs)

class MyFileHandler:
  __handlers = {}
  __counts = collections.Counter()

  def __init__(self, filename):
    self.__filename = filename
    if filename not in self.__handlers:
      self.__handlers[filename] = logging.FileHandler(filename)
    self.__handler = self.__handlers[filename]
    self.__counts[filename] += 1

  def close(self):
    self.__counts[self.__filename] -= 1
    if not self.__counts[self.__filename]:
      self.__handler.close()
      del self.__handlers[self.__filename]

  def __getattr__(self, attr):
    return getattr(self.__handler, attr)

__notgiven = object()

@functools.lru_cache(maxsize=None)
def getlogger(*, module, root, samp, uselogfiles=__notgiven, threshold=__notgiven, mainlog=__notgiven, samplelog=__notgiven):
  if uselogfiles is __notgiven:
    return getlogger(module=module, root=root, samp=samp, uselogfiles=False, threshold=threshold, mainlog=mainlog, samplelog=samplelog)
  if threshold is __notgiven:
    return getlogger(module=module, root=root, samp=samp, uselogfiles=uselogfiles, threshold=logging.DEBUG, mainlog=mainlog, samplelog=samplelog)
  if mainlog is __notgiven:
    return getlogger(module=module, root=root, samp=samp, uselogfiles=uselogfiles, threshold=threshold, mainlog=None, samplelog=samplelog)
  if samplelog is __notgiven:
    return getlogger(module=module, root=root, samp=samp, uselogfiles=uselogfiles, threshold=threshold, mainlog=mainlog, samplelog=None)
  return MyLogger(module, root, samp, uselogfiles=uselogfiles, threshold=threshold, mainlog=mainlog, samplelog=samplelog)
