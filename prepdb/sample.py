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

  def getXMLpolygonannotations(self):
    xmlfile = self.scanfolder/(self.samp+"_"+self.scanfolder.name+"_annotations.polygons.xml")
    annotations = []
    allregions = []
    allvertices = []
    with open(filename, "rb") as f:
      for n, (path, _, node) in enumerate(jxmlease.parse(f, generator="/Annotations/Annotation"), start=1):
        linecolor = f"{node.get_xml_attr('LineColor'):06x}"
        linecolor = linecolor[4:6] + linecolor[2:4] + linecolor[0:2]
        visible = node.get_xml_attr("Visible").lower() == "true"
        name = node.get_xml_attr("Name")
        annotations.append(
          Annotation(
            linecolor=linecolor,
            visible=visible,
            name=name,
            sampleid=0,
            layer=n,
            poly="poly",
          )
        )

        regions = node["Regions"]["Region"]
        if isinstance(regions, jxmlease.XMLDictNode): regions = regions,
        for m, region in enumerate(regions, start=1):
          regionid = 1000*n + m
          vertices = region["Vertices"]["V"]
          if isinstance(vertices, jxmlease.XMLDictNode): vertices = vertices,
          regionvertices = []
          for k, vertex in enumerate(vertices, start=1):
            x = units.Distance(microns=vertex.get_xml_attr("X"))
            y = units.Distance(microns=vertex.get_xml_attr("Y"))
            regionvertices.append(
              Vertex(
                regionid=regionid,
                vid=k,
                x=x,
                y=y,
              )
            )
          allvertices += regionvertices

          isNeg = bool(int(region.get_xml_attr("NegativeROA")))
          if isNeg: regionvertices.reverse()
          polygonvertices = []
          for vertex in regionvertices:
            polygonvertices.append(f"{math.floor(units.pixels(vertex.x))} {math.floor(units.pixels(vertex.y))}")

          allregions.append(
            Region(
              regionid=regionid,
              sampleid=0,
              layer=n,
              rid=m,
              isNeg=isNeg,
              type=region.get_xml_attr("Type"),
              nvert=len(vertices),
              poly="POLYGON ((" + ",".join(polygonvertices) + "))",
            )
          )

    return annotations, allregions, allvertices

  def getqptiff(self):
    qptifffilename = self.scanfolder/(self.samp+"_"+self.scanfolder.name+".qptiff")
    with open(qptifffilename) as f:
      tags = exifread.process_file(f)
    resolutionunit = str(tags["Image ResolutionUnit"])
    xposition = tags["Image XPosition"].values[0]
    xposition = xposition.num / xposition.den
    yposition = tags["Image YPosition"].values[0]
    yposition = yposition.num / yposition.den
    xresolution = tags["Image XResolution"].values[0]
    xresolution = xresolution.num / xresolution.den
    yresolution = tags["Image YResolution"].values[0]
    yresolution = yresolution.num / yresolution.den

    kw = {
      "Pixels/Centimeter": "centimeters",
      "Pixels/Micron": "microns",
    }[resolutionunit]
    xresolution = units.Distance(pixels=xresolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1)
    yresolution = units.Distance(pixels=yresolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1)
    qpscale = xresolution
    xposition = units.Distance(**{kw: xposition}, pscale=qpscale)
    yposition = units.Distance(**{kw: yposition}, pscale=qpscale)
    qptiffcsv = [
      QPTiffCsv(
        SampleID=0,
        SlideID=self.samp,
        ResolutionUnit="Micron",
        XPosition=xposition,
        YPosition=yposition,
        XResolution=xresolution,
        YResolution=yresolution,
        qpscale=qpscale,
        fname=jpgfile.name,
        img="00001234",
      )
    ]

    with PILmaximagepixels(1024**3), PIL.Image.open(qptifffilename) as f:
      raise NotImplementedError

  def getoverlaps(self):
    overlaps = []
    for r1, r2 in itertools.product(self.rectangles, repeat=2):
      if r1 is r2: continue
      if np.all(abs(r1.cxvec - r2.cxvec) < r1.sizevec):
        tag = (-1)**(r1.cx < r2.cx) + 3*(-1)**(r1.cy < r2.cy) + 5
        overlaps.append(
          Overlap(
            n=len(overlaps)+1,
            p1=r1.n,
            p2=r2.n,
            x1=r1.x,
            y1=r1.y,
            x2=r2.x,
            y2=r2.y,
            tag=tag,
            layer=self.layer,
            nclip=self.nclip,
            rectangles=(r1, r2),
            pscale=None,
          )
        )
    return overlaps

  def getconstants(self): raise NotImplementedError
  def writemetadata(self): raise NotImplementedError
