import argparse, fractions, jxmlease, methodtools, numpy as np, PIL, skimage, tifffile
from ..baseclasses.csvclasses import Annotation, Constant, Batch, Polygon, QPTiffCsv, Region, Vertex
from ..baseclasses.overlap import RectangleOverlapCollection
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
            x = int(vertex.get_xml_attr("X")) * self.onemicron
            y = int(vertex.get_xml_attr("Y")) * self.onemicron
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
        if page.imagewidth < 2000:
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
      apscale = firstpage.tags["XResolution"].value
      apscale = float(fractions.Fraction(*apscale))

      kw = {
        tifffile.TIFF.RESUNIT.CENTIMETER: "centimeters",
      }[resolutionunit]
      xresolution = units.Distance(pixels=xresolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1)
      yresolution = units.Distance(pixels=yresolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1)
      qpscale = xresolution
      apscale = units.Distance(pixels=apscale, pscale=1) / units.Distance(**{kw: 1}, pscale=1)

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
          apscale=apscale,
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
  def apscale(self):
    return self.getqptiffcsv()[0].apscale

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
        name='flayers',
        value=self.flayers,
        unit='',
        description='field depth',
        pscale=self.pscale,
      ),
      Constant(
        name='locx',
        value=self.getsamplelocation()[0],
        unit='microns',
        description='xlocation',
        pscale=self.qpscale,
      ),
      Constant(
        name='locy',
        value=self.getsamplelocation()[1],
        unit='microns',
        description='ylocation',
        pscale=self.qpscale,
      ),
      Constant(
        name='locz',
        value=self.getsamplelocation()[2],
        unit='microns',
        description='zlocation',
        pscale=self.qpscale,
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
        name='apscale',
        value=self.apscale,
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
        value=self.nclip * self.onepixel,
        unit='pixels',
        description='pixels to clip off the edge after warping',
        pscale=self.pscale,
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
