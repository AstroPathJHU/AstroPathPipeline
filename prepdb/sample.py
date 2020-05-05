import dateutil, jxmlease, logging, pathlib

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

    with open(self.scanfolder/"BatchID.txt") as f:
      batch = int(f.read())

    R, G = self.getlayout()
    if not R:
      raise ValueError("No layout annotations")
    P = self.getXMLpolygonannotations()
    Q = self.getqptiff()
    qpscale = Q.qpscale
    xposition = Q.xposition
    yposition = Q.yposition

  def getlayout(self):
    r, G, p = self.getXMLplan()
    t = self.getdir()
    raise NotImplementedError

  def getXMLplan(self):
    xmlfile = self.scanfolder/(self.samp+"_"+self.scanfolder.name+"_annotations.xml")
    result = []
    reader = AnnotationXMLReader(xmlfile)
    rectangles = reader.rectangles
    r = self.fixM2(r)
    return r, G, p

  def getdir(self):
    folder = self.scanfolder/"MSI"
    im3s = folder.glob("*.im3")
    result = []
    for im3 in im3s:
      regex = self.samp+r"_\[([0-9]+),([0-9]+)\].im3"
      match = re.match(regex, im3.name)
      if not match:
        raise ValueError(f"Unknown im3 filename {im3}, should match {regex}")
      x = match.group(1)
      y = match.group(2)
      t = os.path.getmtime(im3)
      result.append(
        Rectangle(
          x=x,
          y=y,
          t=t
        )
      )
    result.sort(key=lambda x: x.t)
    return result

  def getXMLpolygonannotations(self): raise NotImplementedError
  def getqptiff(self): raise NotImplementedError
  def getoverlaps(self): raise NotImplementedError
  def getconstants(self): raise NotImplementedError
  def writemetadata(self): raise NotImplementedError
