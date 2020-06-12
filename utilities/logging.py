import logging

def getlogger(module, root, samp, uselogfiles=False):
  logger = logging.getLogger(f"{module}.{root}.{samp}")
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter(f"{samp}, %(message)s, %(asctime)s", "%d-%b-%Y %H:%M:%S")

  for _ in logger.handlers:
    _.close()
  del logger.handlers[:]

  def filter(record):
    levelstoadd = ("INFO", "WARNING", "ERROR")
    if record.levelname in levelstoadd and not record.msg.startswith(record.levelname+": "):
      record.msg = f"{record.levelname}: {record.msg}"
    #if "," in record.msg:
    #  raise ValueError("log messages aren't supposed to have commas:\n\n"+record.msg)
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

    (root/samp/"logfiles").mkdir(exist_ok=True)
    samplehandler = logging.FileHandler(root/samp/"logfiles"/f"{module}.log")
    samplehandler.setFormatter(formatter)
    samplehandler.addFilter(filter)
    samplehandler.setLevel(logging.DEBUG)
    logger.addHandler(samplehandler)

  return logger

dummylogger = logging.getLogger("dummy")
dummylogger.addHandler(logging.NullHandler())
