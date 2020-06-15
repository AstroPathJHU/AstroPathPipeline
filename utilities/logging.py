import dataclasses, logging

@dataclasses.dataclass
class SampleDef:
  SampleID: int
  SlideID: str
  Project: int
  Cohort: int
  Scan: int
  BatchID: int
  isGood: int

  def __bool__(self):
    return bool(self.isGood)

def getlogger(module, root, samp, *, uselogfiles=False):
  if isinstance(samp, SampleDef):
    SampleID = samp.SampleID
    SlideID = samp.SlideID
    Project = samp.Project
    Cohort = samp.Cohort
  else:
    SlideID = samp
    SampleID = Project = Cohort = None

  logger = logging.getLogger(f"{module}.{root}.{Project}.{Cohort}.{SlideID}")
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
    levelstoadd = ("WARNING", "ERROR")
    levelname = record.levelname
    if levelname == "INFO": levelname = "WARNING"
    if record.levelname in levelstoadd and not record.msg.startswith(record.levelname+": "):
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
    mainhandler.setLevel(logging.WARNING)
    logger.addHandler(mainhandler)

    (root/SlideID/"logfiles").mkdir(exist_ok=True)
    samplehandler = logging.FileHandler(root/SlideID/"logfiles"/f"{module}.log")
    samplehandler.setFormatter(formatter)
    samplehandler.addFilter(filter)
    samplehandler.setLevel(logging.INFO)
    logger.addHandler(samplehandler)

  return logger

dummylogger = logging.getLogger("dummy")
dummylogger.addHandler(logging.NullHandler())
