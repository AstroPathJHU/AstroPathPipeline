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

    rectangles, globals = self.getlayout()
    if not R:
      raise ValueError("No layout annotations")
    P = self.getXMLpolygonannotations()
    Q = self.getqptiff()
    qpscale = Q.qpscale
    xposition = Q.xposition
    yposition = Q.yposition
    raise NotImplementedError

  def getlayout(self):
    rectangles, globals, perimeters = self.getXMLplan()
    rectanglefiles = self.getdir()
    maxtimediff = 0
    for r in rectangles:
      rfs = {rf for rf in rectanglefiles if np.all(rf.cxvec == r.cxvec)}
      assert len(rfs) <= 1
      if not rfs:
        raise OSError(f"File {self.samp}_[{r.cx},{r.cy}].im3 (expected from annotations) does not exist")
      maxtimediff = max(maxtimediff, abs(rf.t-r.time))
    if maxtimediff >= datetime.timedelta(seconds=5):
      logger.warning(f"Biggest time difference between annotation and file mtime is {maxtimediff}")
    rectangles.sort(key=lambda x: x.time)
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i
    return rectangles, globals

  def getXMLplan(self):
    xmlfile = self.scanfolder/(self.samp+"_"+self.scanfolder.name+"_annotations.xml")
    result = []
    reader = AnnotationXMLReader(xmlfile)

    rectangles = reader.rectangles
    globals = reader.globals
    perimeters = reader.perimeters
    self.fixM2(rectangles)

    return rectangles, globals, perimeters

  @staticmethod
  def fixM2(rectangles):
    for rectangle in rectangles[:]:
      if "_M2" in rectangle.file:
        duplicates = [r for r in rectangles if r is not rectangle and np.all(r.cxvec == rectangle.cxvec)]
        if not duplicates:
          rectangle.file = rectangle.file.replace("_M2", "")
        for d in duplicates:
          rectangles.remove(d)
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i

  def getdir(self):
    folder = self.scanfolder/"MSI"
    im3s = folder.glob("*.im3")
    result = set()
    for im3 in im3s:
      regex = self.samp+r"_\[([0-9]+),([0-9]+)\].im3"
      match = re.match(regex, im3.name)
      if not match:
        raise ValueError(f"Unknown im3 filename {im3}, should match {regex}")
      x = match.group(1)
      y = match.group(2)
      t = datetime.fromtimestamp(os.path.getmtime(im3))
      result.add(
        RectangleFile(
          cx=x,
          cy=y,
          t=t,
        )
      )
    result.sort(key=lambda x: x.t)
    return result

  def getXMLpolygonannotations(self): raise NotImplementedError
  def getqptiff(self): raise NotImplementedError
  def getoverlaps(self): raise NotImplementedError
  def getconstants(self): raise NotImplementedError
  def writemetadata(self): raise NotImplementedError
