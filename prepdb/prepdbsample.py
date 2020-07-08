import argparse, datetime, fractions, itertools, jxmlease, methodtools, numpy as np, os, pathlib, PIL, re, skimage, tifffile
from ..baseclasses.sample import SampleBase
from ..utilities import units
from ..utilities.misc import floattoint
from .annotationxmlreader import AnnotationXMLReader
from .csvclasses import Annotation, Constant, Batch, Polygon, QPTiffCsv, RectangleFile, Region, Vertex
from .overlap import Overlap

jxmleaseversion = jxmlease.__version__.split(".")
jxmleaseversion = [int(_) for _ in jxmleaseversion[:2]] + list(jxmleaseversion[2:])
if jxmleaseversion < [1, 0, '2dev1']:
  raise ImportError(f"You need jxmleaseversion >= 1.0.2dev1 (your version: {jxmlease.__version__})\n(earlier one has bug in reading vertices, https://github.com/Juniper/jxmlease/issues/16)")

class PrepdbSample(SampleBase):
  @property
  def logmodule(self): return "prepdb"

  @property
  def nclip(self):
    return 8

  @property
  def layer(self):
    return 1

  @methodtools.lru_cache()
  def getbatch(self):
    return [
      Batch(
        Batch=self.BatchID,
        Scan=self.Scan,
        SampleID=self.SampleID,
        Sample=self.SlideID,
      )
    ]

  def writebatch(self):
    self.logger.info("writebatch")
    self.writecsv("batch", self.getbatch())

  @property
  def rectangles(self): return self.getlayout()[0]
  def writerectangles(self):
    self.logger.info("writerectangles")
    self.writecsv("rect", self.rectangles)
  @property
  def globals(self): return self.getlayout()[1]
  def writeglobals(self):
    if not self.globals: return
    self.logger.info("writeglobals")
    self.writecsv("globals", self.globals)

  @methodtools.lru_cache()
  def getlayout(self):
    rectangles, globals, perimeters = self.getXMLplan()
    rectanglefiles = self.getdir()
    maxtimediff = datetime.timedelta(0)
    for r in rectangles:
      rfs = {rf for rf in rectanglefiles if np.all(rf.cxvec == r.cxvec)}
      assert len(rfs) <= 1
      if not rfs:
        cx, cy = units.microns(r.cxvec, pscale=self.tiffpscale)
        raise OSError(f"File {self.SlideID}_[{cx},{cy}].im3 (expected from annotations) does not exist")
      rf = rfs.pop()
      maxtimediff = max(maxtimediff, abs(rf.t-r.t))
    if maxtimediff >= datetime.timedelta(seconds=5):
      self.logger.warning(f"Biggest time difference between annotation and file mtime is {maxtimediff}")
    rectangles.sort(key=lambda x: x.t)
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i
    if not rectangles:
      raise ValueError("No layout annotations")
    return rectangles, globals

  @methodtools.lru_cache()
  def getXMLplan(self):
    xmlfile = self.scanfolder/(self.SlideID+"_"+self.scanfolder.name+"_annotations.xml")
    reader = AnnotationXMLReader(xmlfile, pscale=self.tiffpscale)

    rectangles = reader.rectangles
    globals = reader.globals
    perimeters = reader.perimeters
    self.fixM2(rectangles)
    self.fixrectanglefilenames(rectangles)

    return rectangles, globals, perimeters

  def fixM2(self, rectangles):
    for rectangle in rectangles[:]:
      if "_M2" in rectangle.file:
        duplicates = [r for r in rectangles if r is not rectangle and np.all(r.cxvec == rectangle.cxvec)]
        if not duplicates:
          rectangle.file = rectangle.file.replace("_M2", "")
        for d in duplicates:
          rectangles.remove(d)
        self.logger.warningglobal(f"{rectangle.file} has _M2 in the name.  {len(duplicates)} other duplicate rectangles.")
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i

  def fixrectanglefilenames(self, rectangles):
    for r in rectangles:
      expected = self.SlideID+f"_[{floattoint(units.microns(r.cx, pscale=r.pscale), atol=1e-10):d},{floattoint(units.microns(r.cy, pscale=r.pscale), atol=1e-10):d}].im3"
      actual = r.file
      if expected != actual:
        self.logger.warningglobal(f"rectangle at ({r.cx}, {r.cy}) has the wrong filename {actual}.  Changing it to {expected}.")
      r.file = expected

  @methodtools.lru_cache()
  def getdir(self):
    folder = self.scanfolder/"MSI"
    im3s = folder.glob("*.im3")
    result = []
    for im3 in im3s:
      regex = self.SlideID+r"_\[([0-9]+),([0-9]+)\].im3"
      match = re.match(regex, im3.name)
      if not match:
        raise ValueError(f"Unknown im3 filename {im3}, should match {regex}")
      x = units.Distance(microns=int(match.group(1)), pscale=self.tiffpscale)
      y = units.Distance(microns=int(match.group(2)), pscale=self.tiffpscale)
      t = datetime.datetime.fromtimestamp(os.path.getmtime(im3)).astimezone()
      result.append(
        RectangleFile(
          cx=x,
          cy=y,
          t=t,
          pscale=self.tiffpscale,
        )
      )
    result.sort(key=lambda x: x.t)
    return result

  @methodtools.lru_cache()
  def getXMLpolygonannotations(self):
    xmlfile = self.scanfolder/(self.SlideID+"_"+self.scanfolder.name+".annotations.polygons.xml")
    if not xmlfile.exists():
      return [], [], []
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
            x = units.Distance(microns=int(vertex.get_xml_attr("X")), pscale=self.tiffpscale)
            y = units.Distance(microns=int(vertex.get_xml_attr("Y")), pscale=self.tiffpscale)
            regionvertices.append(
              Vertex(
                regionid=regionid,
                vid=k,
                x=x,
                y=y,
                pscale=self.tiffpscale,
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
              pscale=self.tiffpscale,
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
    self.logger.info("writeannotations")
    self.writecsv("annotations", self.annotations, rowclass=Annotation)
  def writeregions(self):
    self.logger.info("writeregions")
    self.writecsv("regions", self.regions, rowclass=Region)
  def writevertices(self):
    self.logger.info("writevertices")
    self.writecsv("vertices", self.vertices, rowclass=Vertex)

  @property
  def qptifffilename(self): return self.scanfolder/(self.SlideID+"_"+self.scanfolder.name+".qptiff")
  @property
  def jpgfilename(self): return self.dbload/(self.SlideID+"_qptiff.jpg")

  @methodtools.lru_cache()
  def getqptiffcsvandimage(self):
    with tifffile.TiffFile(self.qptifffilename) as f:
      layeriterator = iter(enumerate(f.pages))
      for qplayeridx, page in layeriterator:
        #get to after the small RGB one
        if len(page.shape) == 3:
          break
      else:
        raise ValueError("Unexpected qptiff layout: expected to find an RGB layer (with a 3D array shape).  Array shapes:\n" + "\n".join(f"  {page.shape}" for page in f.pages))
      for qplayeridx, page in layeriterator:
        if page.imagewidth < 4000:
          break
      else:
        raise ValueError("Unexpected qptiff layout: expected layer with width < 4000 sometime after the first RGB layer (with a 3D array shape).  Shapes and widths:\n" + "\n".join(f"  {page.shape:20} {page.imagewidth:10}" for page in f.pages))

      firstpage = f.pages[0]
      resolutionunit = firstpage.tags["ResolutionUnit"].value
      xposition = firstpage.tags["XPosition"].value
      xposition = float(fractions.Fraction(*xposition))
      yposition = firstpage.tags["YPosition"].value
      yposition = float(fractions.Fraction(*yposition))
      xresolution = page.tags["XResolution"].value
      xresolution = float(fractions.Fraction(*xresolution))
      yresolution = page.tags["YResolution"].value
      yresolution = float(fractions.Fraction(*yresolution))

      kw = {
        tifffile.TIFF.RESUNIT.CENTIMETER: "centimeters",
      }[resolutionunit]
      xresolution = units.Distance(pixels=xresolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1)
      yresolution = units.Distance(pixels=yresolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1)
      qpscale = xresolution

      xposition = units.Distance(**{kw: xposition}, pscale=qpscale)
      yposition = units.Distance(**{kw: yposition}, pscale=qpscale)
      qptiffcsv = [
        QPTiffCsv(
          SampleID=0,
          SlideID=self.SlideID,
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

      shape = *f.pages[qplayeridx].shape, 3
      finalimg = np.zeros(shape)

      for i in range(qplayeridx, qplayeridx+5):
        img = f.pages[i].asarray()
        finalimg += np.tensordot(img, mix[:,i-qplayeridx], axes=0)
#    finalimg /= np.max(finalimg)
    finalimg[finalimg > 1] = 1
    qptiffimg = skimage.img_as_ubyte(finalimg)
    return qptiffcsv, PIL.Image.fromarray(qptiffimg)

  def getqptiffcsv(self):
    return self.getqptiffcsvandimage()[0]
  def getqptiffimage(self):
    return self.getqptiffcsvandimage()[1]

  def writeqptiffcsv(self):
    self.logger.info("writeqptiffcsv")
    self.writecsv("qptiff", self.getqptiffcsv())

  def writeqptiffjpg(self):
    self.logger.info("writeqptiffjpg")
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
            pscale=self.tiffpscale,
            readingfromfile=False,
          )
        )
    return overlaps

  def writeoverlaps(self):
    self.logger.info("writeoverlaps")
    self.writecsv("overlap", self.getoverlaps())

  def getconstants(self):
    constants = [
      Constant(
        name='fwidth',
        value=self.tiffwidth,
        unit='pixels',
        description='field width',
        pscale=self.tiffpscale,
      ),
      Constant(
        name='fheight',
        value=self.tiffheight,
        unit='pixels',
        description='field height',
        pscale=self.tiffpscale,
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
        value=self.tiffpscale,
        unit='pixels/micron',
        description='scale of the HPF images',
      ),
      Constant(
        name='nclip',
        value=units.Distance(pixels=self.nclip, pscale=self.tiffpscale),
        unit='pixels',
        description='pixels to clip off the edge after warping',
        pscale=self.tiffpscale,
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
    self.logger.info("writeconstants")
    self.writecsv("constants", self.getconstants())

  def writemetadata(self):
    self.dbload.mkdir(parents=True, exist_ok=True)
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

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("root")
  p.add_argument("samp")
  p.add_argument("--units", type=units.setup)
  args = p.parse_args()
  kwargs = {"root": args.root, "samp": args.samp}
  s = PrepdbSample(**kwargs)
  s.writemetadata()
