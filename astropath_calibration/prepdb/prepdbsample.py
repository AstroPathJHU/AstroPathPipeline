import argparse, fractions, jxmlease, methodtools, numpy as np, PIL, skimage, tifffile
from ..baseclasses.csvclasses import Annotation, Constant, Batch, Polygon, QPTiffCsv, Region, Vertex
from ..baseclasses.overlap import RectangleOverlapCollection
from ..baseclasses.qptiff import QPTiff
from ..baseclasses.sample import DbloadSampleBase, XMLLayoutReader
from ..utilities import units

class PrepdbSampleBase(XMLLayoutReader, RectangleOverlapCollection):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, checkim3s=True, **kwargs)

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

  @property
  def rectangles(self): return self.getrectanglelayout()

  @property
  def globals(self): return self.getXMLplan()[1]

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

  @property
  def jpgfilename(self): return self.dbload/(self.SlideID+"_qptiff.jpg")

  @methodtools.lru_cache()
  def getqptiffcsvandimage(self):
    with QPTiff(self.qptifffilename) as f:
      for zoomlevel in f.zoomlevels:
        if zoomlevel[0].imagewidth < 4000:
          break

      resolutionunit = zoomlevel.tags["ResolutionUnit"]
      xposition = zoomlevel.tags["XPosition"]
      xposition = float(fractions.Fraction(*xposition))
      yposition = zoomlevel.tags["YPosition"]
      yposition = float(fractions.Fraction(*yposition))
      xresolution = zoomlevel.tags["XResolution"]
      xresolution = float(fractions.Fraction(*xresolution))
      yresolution = zoomlevel.tags["YResolution"]
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

      shape = *zoomlevel.shape, 3
      finalimg = np.zeros(shape)

      for i, page in enumerate(zoomlevel[:5]):
        img = page.asarray()
        finalimg += np.tensordot(img, mix[:,i], axes=0)
#    finalimg /= np.max(finalimg)
    finalimg[finalimg > 1] = 1
    qptiffimg = skimage.img_as_ubyte(finalimg)
    return qptiffcsv, PIL.Image.fromarray(qptiffimg)

  def getqptiffcsv(self):
    return self.getqptiffcsvandimage()[0]
  def getqptiffimage(self):
    return self.getqptiffcsvandimage()[1]

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

  @property
  def overlaps(self):
    return self.getoverlaps()

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

class PrepdbSample(PrepdbSampleBase, DbloadSampleBase):
  def writebatch(self):
    self.logger.info("writebatch")
    self.writecsv("batch", self.getbatch())

  def writerectangles(self):
    self.logger.info("writerectangles")
    self.writecsv("rect", self.rectangles)

  def writeglobals(self):
    if not self.globals: return
    self.logger.info("writeglobals")
    self.writecsv("globals", self.globals)

  def writeannotations(self):
    self.logger.info("writeannotations")
    self.writecsv("annotations", self.annotations, rowclass=Annotation)

  def writeregions(self):
    self.logger.info("writeregions")
    self.writecsv("regions", self.regions, rowclass=Region)

  def writevertices(self):
    self.logger.info("writevertices")
    self.writecsv("vertices", self.vertices, rowclass=Vertex)

  def writeqptiffcsv(self):
    self.logger.info("writeqptiffcsv")
    self.writecsv("qptiff", self.getqptiffcsv())

  def writeqptiffjpg(self):
    self.logger.info("writeqptiffjpg")
    img = self.getqptiffimage()
    img.save(self.jpgfilename)

  def writeoverlaps(self):
    self.logger.info("writeoverlaps")
    self.writecsv("overlap", self.getoverlaps())

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

def main(args=None):
  p = argparse.ArgumentParser()
  p.add_argument("root")
  p.add_argument("samp")
  p.add_argument("--units", type=units.setup)
  args = p.parse_args(args=args)
  kwargs = {"root": args.root, "samp": args.samp}
  s = PrepdbSample(**kwargs)
  s.writemetadata()

if __name__ == "__main__":
  main()
