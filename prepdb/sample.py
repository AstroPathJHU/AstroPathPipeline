import logging, pathlib

logger = logging.getLogger("prepdb")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s, %(funcName)s, %(asctime)s"))
logger.addHandler(handler)

class Sample:
  def __init__(self, root, samp):
    self.root = pathlib.Path(root)
    self.samp = samp

  @property
  def dest(self):
    return self.root/self.samp/"dbload"

  @property
  def scanfolder(self):
    im3folder = self.root/self.samp/"im3"
    return max(im3folder.glob("Scan*/"), key=lambda folder: int(folder.name.replace("Scan", "")))

  @property
  def nclip(self):
    return 8

  @property
  def layer(self):
    return 1

  def getmetadata(self): raise NotImplementedError
  def getoverlaps(self): raise NotImplementedError
  def getconstants(self): raise NotImplementedError
  def writemetadata(self): raise NotImplementedError
