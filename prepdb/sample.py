import datetime, exifreader, itertools, jxmlease, logging, methodtools, numpy as np, os, pathlib, PIL, re, skimage
from ..utilities import units
from ..utilities.tableio import writetable
from .annotationxmlreader import AnnotationXMLReader
from .csvclasses import Annotation, Constant, Batch, Polygon, QPTiffCsv, RectangleFile, Region, Vertex
from .overlap import Overlap

logger = logging.getLogger("prepdb")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s, %(funcName)s, %(asctime)s"))
logger.addHandler(handler)

jxmleaseversion = jxmlease.__version__.split(".")
jxmleaseversion = [int(_) for _ in jxmleaseversion[:2]] + list(jxmleaseversion[2:])
if jxmleaseversion < [1, 0, '2dev1']:
  raise ImportError(f"You need jxmleaseversion >= 1.0.2dev1 (your version: {jxmlease.__version__})\n(earlier one has bug in reading vertices, https://github.com/Juniper/jxmlease/issues/16)")

class Sample:
  def __init__(self, root, samp, *, dest=None):
    self.root = pathlib.Path(root)
    self.samp = samp
    if dest is None:
      dest = self.root/self.samp/"dbload"
    self.__dest = pathlib.Path(dest)

  @property
  def dest(self): return self.__dest

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

  @methodtools.lru_cache()
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

  def writebatch(self):
    logger.info(self.samp)
    writetable(self.dest/(self.samp+"_batch.csv"), self.getbatch())

  @methodtools.lru_cache()
  def getcomponenttiffinfo(self):
    componenttifffilename = next(self.componenttiffsfolder.glob(self.samp+"*_component_data.tif"))
    with PIL.Image.open(componenttifffilename) as tiff:
      dpi = set(tiff.info["dpi"])
      if len(dpi) != 1: raise ValueError(f"Multiple different dpi values {dpi}")
      pscale = dpi.pop() / 2.54 / 10000
      fwidth, fheight = units.distances(pixels=tiff.size, pscale=pscale, power=1)
    return pscale, fwidth, fheight

  @property
  def pscale(self): return self.getcomponenttiffinfo()[0]
  @property
  def fwidth(self): return self.getcomponenttiffinfo()[1]
  @property
  def fheight(self): return self.getcomponenttiffinfo()[2]

  @property
  def rectangles(self): return self.getlayout()[0]
  def writerectangles(self):
    logger.info(self.samp)
    writetable(self.dest/(self.samp+"_rect.csv"), self.rectangles)
  @property
  def globals(self): return self.getlayout()[1]
  def writeglobals(self):
    if not self.globals: return
    logger.info(self.samp)
    writetable(self.dest/(self.samp+"_globals.csv"), self.globals)

  @methodtools.lru_cache()
  def getlayout(self):
    rectangles, globals, perimeters = self.getXMLplan()
    rectanglefiles = self.getdir()
    maxtimediff = datetime.timedelta(0)
    for r in rectangles:
      rfs = {rf for rf in rectanglefiles if np.all(rf.cxvec == r.cxvec)}
      assert len(rfs) <= 1
      if not rfs:
        raise OSError(f"File {self.samp}_[{r.cx},{r.cy}].im3 (expected from annotations) does not exist")
      rf = rfs.pop()
      maxtimediff = max(maxtimediff, abs(rf.t-r.t))
    if maxtimediff >= datetime.timedelta(seconds=5):
      logger.warning(f"Biggest time difference between annotation and file mtime is {maxtimediff}")
    rectangles.sort(key=lambda x: x.t)
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i
    if not rectangles:
      raise ValueError("No layout annotations")
    return rectangles, globals

  @methodtools.lru_cache()
  def getXMLplan(self):
    xmlfile = self.scanfolder/(self.samp+"_"+self.scanfolder.name+"_annotations.xml")
    reader = AnnotationXMLReader(xmlfile, pscale=self.pscale)

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
    result = []
    for im3 in im3s:
      regex = self.samp+r"_\[([0-9]+),([0-9]+)\].im3"
      match = re.match(regex, im3.name)
      if not match:
        raise ValueError(f"Unknown im3 filename {im3}, should match {regex}")
      x = units.Distance(microns=int(match.group(1)), pscale=self.pscale)
      y = units.Distance(microns=int(match.group(2)), pscale=self.pscale)
      t = datetime.datetime.fromtimestamp(os.path.getmtime(im3)).astimezone()
      result.append(
        RectangleFile(
          cx=x,
          cy=y,
          t=t,
          pscale=self.pscale,
        )
      )
    result.sort(key=lambda x: x.t)
    return result

  @methodtools.lru_cache()
  def getXMLpolygonannotations(self):
    xmlfile = self.scanfolder/(self.samp+"_"+self.scanfolder.name+".annotations.polygons.xml")
    annotations = []
    allregions = []
    allvertices = []
    with open(xmlfile, "rb") as f:
      for n, (path, _, node) in enumerate(jxmlease.parse(f, generator="/Annotations/Annotation"), start=1):
        color = f"{int(node.get_xml_attr('LineColor')):06X}"
        color = color[4:6] + color[2:4] + color[0:2]
        visible = node.get_xml_attr("Visible").lower() == "true"
        name = node.get_xml_attr("Name")
        annotations.append(
          Annotation(
            color=color,
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
            x = units.Distance(microns=int(vertex.get_xml_attr("X")), pscale=self.pscale)
            y = units.Distance(microns=int(vertex.get_xml_attr("Y")), pscale=self.pscale)
            regionvertices.append(
              Vertex(
                regionid=regionid,
                vid=k,
                x=x,
                y=y,
                pscale=self.pscale,
              )
            )
          allvertices += regionvertices

          isNeg = bool(int(region.get_xml_attr("NegativeROA")))
          if isNeg: regionvertices.reverse()
          polygonvertices = []
          for vertex in regionvertices:
            polygonvertices.append(vertex)

          allregions.append(
            Region(
              regionid=regionid,
              sampleid=0,
              layer=n,
              rid=m,
              isNeg=isNeg,
              type=region.get_xml_attr("Type"),
              nvert=len(vertices),
              poly=Polygon(*polygonvertices),
              pscale=self.pscale,
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
    logger.info(self.samp)
    writetable(self.dest/(self.samp+"_annotations.csv"), self.annotations)
  def writeregions(self):
    logger.info(self.samp)
    writetable(self.dest/(self.samp+"_regions.csv"), self.regions)
  def writevertices(self):
    logger.info(self.samp)
    writetable(self.dest/(self.samp+"_vertices.csv"), self.vertices)

  @property
  def qptifffilename(self): return self.scanfolder/(self.samp+"_"+self.scanfolder.name+".qptiff")
  @property
  def jpgfilename(self): return self.dest/(self.samp+"_qptiff.jpg")

  @methodtools.lru_cache()
  def getqptiffcsvandimage(self):
    with open(self.qptifffilename, "rb") as f:
      tags = exifreader.process_file(f)

    layerids = [k.replace(" ImageWidth", "") for k in tags if "ImageWidth" in k]
    for qplayeridx, qplayerid in enumerate(layerids):
      if qplayeridx < 6:
        continue
      if tags[qplayerid+" ImageWidth"].values[0] < 4000:
        break
    else:
      raise ValueError("Unexpected qptiff layout: expected layer with width < 4000 sometime after the 7th.  Widths:\n" + "\n".join(f"  {qplayerid} {tags[qplayerid+' ImageWidth'].values[0]:d}" for qplayerid in layerids))

    resolutionunit = str(tags["Image ResolutionUnit"])
    xposition = tags["Image XPosition"].values[0]
    xposition = float(xposition)
    yposition = tags["Image YPosition"].values[0]
    yposition = float(yposition)
    xresolution = tags[qplayerid + " XResolution"].values[0]
    xresolution = float(xresolution)
    yresolution = tags[qplayerid + " YResolution"].values[0]
    yresolution = float(yresolution)

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
        XResolution=xresolution * 10000,
        YResolution=yresolution * 10000,
        qpscale=qpscale,
        fname=self.jpgfilename.name,
        img="00001234",
        pscale=qpscale,
      )
    ]

    mix = np.array([
      [0.0, 0.0, 1.0, 1.0, 1.0],
      [0.0, 1.0, 1.0, 0.5, 0.0],
      [1.0, 0.0, 0.0, 0.0, 0.0],
    ])/120

    with open(self.qptifffilename, "rb") as f, PIL.Image.open(f) as imgs:
      iterator = PIL.ImageSequence.Iterator(imgs)
      shape = *reversed(iterator[qplayeridx].size), 3
      finalimg = np.zeros(shape)

      for i in range(qplayeridx, qplayeridx+5):
        img = iterator[i]
        try:
          img.getdata()
        except OSError as e:
          if str(e) == "-2":
            qptiffimage = ImportError("Probably you're on Windows and have a buggy version of libtiff.\nSee https://github.com/python-pillow/Pillow/issues/4237\nTry this, but it may be painful to make it work:\n  conda install -c conda-forge libtiff=4.1.0=h885aae3_4")
            return qptiffcsv, qptiffimage
          raise
        finalimg += np.tensordot(np.asarray(img), mix[:,i-qplayeridx], axes=0)
#    finalimg /= np.max(finalimg)
    finalimg[finalimg > 1] = 1
    qptiffimg = skimage.img_as_ubyte(finalimg)
    return qptiffcsv, PIL.Image.fromarray(qptiffimg)

  def getqptiffcsv(self):
    return self.getqptiffcsvandimage()[0]
  def getqptiffimage(self):
    img = self.getqptiffcsvandimage()[1]
    if isinstance(img, Exception): raise img
    return img

  def writeqptiffcsv(self):
    logger.info(self.samp)
    writetable(self.dest/(self.samp+"_qptiff.csv"), self.getqptiffcsv())

  def writeqptiffjpg(self):
    logger.info(self.samp)
    img = self.getqptiffimage()
    img.save(self.jpgfilename)

  @property
  def xposition(self):
    return self.getqptiffcsv()[0].XPosition
  @property
  def yposition(self):
    return self.getqptiffcsv()[0].YPosition
  @property
  def xresolution(self):
    return self.getqptiffcsv()[0].XResolution
  @property
  def yresolution(self):
    return self.getqptiffcsv()[0].YResolution
  @property
  def qpscale(self):
    return self.getqptiffcsv()[0].qpscale

  @methodtools.lru_cache()
  def getoverlaps(self):
    overlaps = []
    for r1, r2 in itertools.product(self.rectangles, repeat=2):
      if r1 is r2: continue
      if np.all(abs(r1.cxvec - r2.cxvec) < r1.shape):
        tag = int(np.sign(r1.cx-r2.cx)) + 3*int(np.sign(r1.cy-r2.cy)) + 5
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
            pscale=self.pscale,
            readingfromfile=False,
          )
        )
    return overlaps

  def writeoverlaps(self):
    logger.info(self.samp)
    writetable(self.dest/(self.samp+"_overlap.csv"), self.getoverlaps())

  def getconstants(self):
    constants = [
      Constant(
        name='fwidth',
        value=self.fwidth,
        unit='pixels',
        description='field width',
        pscale=self.pscale,
      ),
      Constant(
        name='fheight',
        value=self.fheight,
        unit='pixels',
        description='field height',
        pscale=self.pscale,
      ),
      Constant(
        name='xposition',
        value=self.xposition,
        unit='microns',
        description='slide x offset',
        pscale=self.qpscale,
      ),
      Constant(
        name='yposition',
        value=self.yposition,
        unit='microns',
        description='slide y offset',
        pscale=self.qpscale,
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
        value=units.Distance(pixels=self.nclip, pscale=self.pscale),
        unit='pixels',
        description='pixels to clip off the edge after warping',
        pscale=self.pscale,
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
    logger.info(self.samp)
    writetable(self.dest/(self.samp+"_constants.csv"), self.getconstants())

  def writemetadata(self):
    self.writeannotations()
    self.writebatch()
    self.writeconstants()
    self.writeglobals()
    self.writeoverlaps()
    self.writeqptiffcsv()
    self.writeqptiffjpg()
    self.writerectangles()
    self.writeregions()
    self.writevertices()

