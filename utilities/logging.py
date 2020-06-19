import logging

from ..baseclasses.sample import SampleDef

class MyLogger(object):
  def __init__(self, *args, module=None, **kwargs):
    self.logger = logging.getLogger(*args, **kwargs)
    self.module = module
  def __enter__(self):
    self.logger.critical(self.module)
    return self
  def __exit__(self, *exc):
    self.logger.info(f"end {self.module}")
  def __getattr__(self, attr):
    return getattr(self.logger, attr)
  def warningglobal(self, *args, **kwargs):
    return self.logger.log(logging.WARNING+1, *args, **kwargs)

def getlogger(module, root, samp, *, uselogfiles=False):
  if isinstance(samp, SampleDef):
    SampleID = samp.SampleID
    SlideID = samp.SlideID
    Project = samp.Project
    Cohort = samp.Cohort
  else:
    SlideID = samp
    SampleID = Project = Cohort = None

  logger = MyLogger(f"{module}.{root}.{Project}.{Cohort}.{SlideID}", module=module)
  logger.setLevel(logging.DEBUG)

  if uselogfiles and Project is None:
    raise ValueError("Have to give a SampleDef object when writing to log files")
  formatter = logging.Formatter(
    ";".join(f"{_}" for _ in (Project, Cohort, SampleID, SlideID, "%(message)s", "%(asctime)s") if _ is not None),
    "%d-%b-%Y %H:%M:%S",
  )

  for _ in logger.handlers:
    _.close()
  del logger.handlers[:]

  def filter(record):
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

  printhandler = logging.StreamHandler()
  printhandler.setFormatter(formatter)
  printhandler.addFilter(filter)
  printhandler.setLevel(logging.DEBUG)
  logger.addHandler(printhandler)

  if uselogfiles:
    (root/"logfiles").mkdir(exist_ok=True)
    mainhandler = logging.FileHandler(root/"logfiles"/f"{module}.log")
    mainhandler.setFormatter(formatter)
    mainhandler.addFilter(filter)
    mainhandler.setLevel(logging.WARNING+1)
    logger.addHandler(mainhandler)

    (root/SlideID/"logfiles").mkdir(exist_ok=True)
    samplehandler = logging.FileHandler(root/SlideID/"logfiles"/f"{SlideID}-{module}.log")
    samplehandler.setFormatter(formatter)
    samplehandler.addFilter(filter)
    samplehandler.setLevel(logging.INFO)
    logger.addHandler(samplehandler)

  return logger

dummylogger = logging.getLogger("dummy")
dummylogger.addHandler(logging.NullHandler())
