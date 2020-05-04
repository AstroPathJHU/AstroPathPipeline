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
  def componenttiffsfolder(self):
    return self.root/self.samp/"inform_data"/"Component_Tiffs"

  @property
  def nclip(self):
    return 8

  @property
  def layer(self):
    return 1

  def getmetadata(self):
    componenttifffilename = next(self.componenttiffsfolder.glob(self.samp+"*_component_data.tif"))
    with PIL.Image.open(componenttifffilename) as tiff:
      pscale = tiff.info["dpi"] / 2.54
      fwidth, fheight = units.distances(pixels=tiff.size, pscale=pscale, power=1)
    batch = self.getbatch()
    R, G = self.getlayout()
    if not R:
      raise ValueError("No layout annotations")
    P = self.getXMLpolygonannotations()
    Q = self.getqptiff()
    qpscale = Q.qpscale
    xposition = Q.xposition
    yposition = Q.yposition

  def getbatch(self): raise NotImplementedError
  def getlayout(self): raise NotImplementedError
  def getXMLpolygonannotations(self): raise NotImplementedError
  def getoverlaps(self): raise NotImplementedError
  def getconstants(self): raise NotImplementedError
  def writemetadata(self): raise NotImplementedError
