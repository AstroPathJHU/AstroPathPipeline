import dataclasses, datetime, exifread, itertools, jxmlease, logging, math, methodtools, numpy as np, os, pathlib, PIL, re
from ..utilities import units
from ..utilities.misc import PILmaximagepixels
from ..utilities.tableio import writetable
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield
from ..alignment.overlap import Overlap
from .annotationxmlreader import AnnotationXMLReader

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

  def getbatch(self):
    with open(self.scanfolder/"BatchID.txt") as f:
      return [
        Batch(
          Batch=int(f.read()),
          Scan=int(self.scanfolder.name.replace("Scan", "")),
          SampleID=0,
          Sample=self.samp,
        )
      ]

  @methodtools.lru_cache()
  def getcomponenttiffinfo(self):
    componenttifffilename = next(self.componenttiffsfolder.glob(self.samp+"*_component_data.tif"))
    with PIL.Image.open(componenttifffilename) as tiff:
      pscale = tiff.info["dpi"] / 2.54
      fwidth, fheight = units.distances(pixels=tiff.size, pscale=pscale, power=1)
    return pscale, fwidth, fheight

  @property
  def pscale(self): return self.getcomponenttiffinfo()[0]
  @property
  def fwidth(self): return self.getcomponenttiffinfo()[1]
  @property
  def fheight(self): return self.getcomponenttiffinfo()[2]

  @property
  def globals(self): return self.getlayout()[0]
  def writeglobals(self):
    writetable(self.dest/(self.samp+"_globals.csv"), self.globals)
  @property
  def rectangles(self): return self.getlayout()[1]
  def writerectangles(self):
    writetable(self.dest/(self.samp+"_rect.csv"), self.rectangles)

  @methodtools.lru_cache()
  def getlayout(self):
    rectangles, globals, perimeters = self.getXMLplan()
    rectanglefiles = self.getdir()
    maxtimediff = 0
    for r in rectangles:
      rfs = {rf for rf in rectanglefiles if np.all(rf.cxvec == r.cxvec)}
      assert len(rfs) <= 1
      if not rfs:
        raise OSError(f"File {self.samp}_[{r.cx},{r.cy}].im3 (expected from annotations) does not exist")
      rf = rfs.pop()
      maxtimediff = max(maxtimediff, abs(rf.t-r.time))
    if maxtimediff >= datetime.timedelta(seconds=5):
      logger.warning(f"Biggest time difference between annotation and file mtime is {maxtimediff}")
    rectangles.sort(key=lambda x: x.time)
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i
    if not rectangles:
      raise ValueError("No layout annotations")
    return rectangles, globals

  @methodtools.lru_cache()
  def getXMLplan(self):
    xmlfile = self.scanfolder/(self.samp+"_"+self.scanfolder.name+"_annotations.xml")
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

  @methodtools.lru_cache()
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

  @methodtools.lru_cache()
  def getXMLpolygonannotations(self):
    xmlfile = self.scanfolder/(self.samp+"_"+self.scanfolder.name+"_annotations.polygons.xml")
    annotations = []
    allregions = []
    allvertices = []
    with open(xmlfile, "rb") as f:
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

  @property
  def annotations(self): return self.getXMLpolygonannotations()[0]
  @property
  def regions(self): return self.getXMLpolygonannotations()[1]
  @property
  def vertices(self): return self.getXMLpolygonannotations()[2]

  def writeannotations(self):
    writetable(self.dest/(self.samp+"_annotations.csv"), self.annotations)
  def writeregions(self):
    writetable(self.dest/(self.samp+"_regions.csv"), self.regions)
  def writevertices(self):
    writetable(self.dest/(self.samp+"_vertices.csv"), self.vertices)

  @property
  def qptifffilename(self): return self.scanfolder/(self.samp+"_"+self.scanfolder.name+".qptiff")
  @property
  def jpgfilename(self): return self.dest/(self.samp+"_qptiff.jpg")

  @methodtools.lru_cache()
  def getqptiffcsv(self):
    with open(self.qptifffilename) as f:
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
    return [
      QPTiffCsv(
        SampleID=0,
        SlideID=self.samp,
        ResolutionUnit="Micron",
        XPosition=xposition,
        YPosition=yposition,
        XResolution=xresolution,
        YResolution=yresolution,
        qpscale=qpscale,
        fname=self.jpgfilename.name,
        img="00001234",
      )
    ]

  def writeqptiffcsv(self):
    writetable(self.dest/(self.samp+"_qptiff.csv"), self.getqptiffcsv())

  def writeqptiffjpg(self):
    raise NotImplementedError
    with PILmaximagepixels(1024**3), PIL.Image.open(self.qptifffilename) as f:
      f

  @methodtools.lru_cache()
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

  def writeoverlaps(self):
    writetable(self.dest/(self.samp+"_overlap.csv"), self.getoverlaps())

  def getconstants(self):
    constants = [
      Constant(
        name='fwidth',
        value=self.fwidth,
        unit='pixels',
        description='field width',
      ),
      Constant(
        name='fheight',
        value=self.fheight,
        unit='pixels',
        description='field height',
      ),
      Constant(
        name='xposition',
        value=self.xposition,
        unit='microns',
        description='slide x offset',
      ),
      Constant(
        name='yposition',
        value=self.yposition,
        unit='microns',
        description='slide y offset',
      ),
      Constant(
        name='qpscale',
        value=self.qpscale,
        unit='pixels/micron',
        description='scale of the QPTIFF image',
      ),
      Constant(
        name='pscale',
        value=self.pscale,
        unit='pixels/micron',
        description='scale of the HPF images',
      ),
      Constant(
        name='nclip',
        value=self.nclip,
        unit='pixels',
        description='pixels to clip off the edge after warping',
      ),
      Constant(
        name='layer',
        value=self.layer,
        unit='layer number',
        description='which layer to use from the im3 to align',
      ),
    ]
    return constants

  def writeconstants(self):
    writetable(self.dest/(self.samp+"_constants.csv"), self.getconstants())

  def writemetadata(self):
    self.writeconstants()
    self.writeoverlaps()
    self.writeqptiffcsv()
    self.writeqptiffjpg()
    self.writeannotations()
    self.writeregions()
    self.writevertices()
    self.writeglobals()
    self.writerectangles()

@dataclasses.dataclass
class Batch:
  SampleID: int
  Sample: str
  Scan: int
  Batch: int

@dataclasses.dataclass
class QPTiffCsv(DataClassWithDistances):
  pixelsormicrons = "microns"

  SampleID: int
  SlideID: str
  ResolutionUnit: str
  XPosition: distancefield(pixelsormicrons=pixelsormicrons)
  YPosition: distancefield(pixelsormicrons=pixelsormicrons)
  XResolution: float
  YResolution: float
  qpscale: float
  fname: str
  img: str

class Constant:
  def intorfloat(string):
    assert isinstance(string, str)
    try: return int(string)
    except ValueError: return float(string)

  name: str
  value: float = dataclasses.field(metadata={"readfunction": intorfloat})
  unit: str
  description: str
