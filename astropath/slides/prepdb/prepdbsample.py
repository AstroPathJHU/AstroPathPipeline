import argparse, methodtools, numpy as np, PIL, skimage
from ...baseclasses.annotationpolygonxmlreader import XMLPolygonAnnotationReader
from ...baseclasses.csvclasses import Annotation, Constant, Batch, ExposureTime, QPTiffCsv, Region, Vertex
from ...baseclasses.overlap import RectangleOverlapCollection
from ...baseclasses.qptiff import QPTiff
from ...baseclasses.sample import DbloadSampleBase, WorkflowSample, XMLLayoutReader
from ...baseclasses.workflowdependency import ShredXML
from ...utilities import units

class PrepDbSampleBase(XMLLayoutReader, RectangleOverlapCollection, WorkflowSample, units.ThingWithQpscale, units.ThingWithApscale):
  """
  The prepdb stage of the pipeline extracts metadata for a sample from the `.xml` files
  and writes it out to `.csv` files.
  For more information, see README.md in this folder.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, checkim3s=True, **kwargs)

  @classmethod
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
  def exposuretimes(self):
    return [
      ExposureTime(
        n=r.n,
        cx=r.cx,
        cy=r.cy,
        layer=layer,
        exp=exposuretime,
        pscale=self.pscale,
      )
      for r in self.rectangles
      for layer, exposuretime in enumerate(r.allexposuretimes, start=1)
    ]

  @property
  def globals(self): return self.getXMLplan()[1]

  @methodtools.lru_cache()
  def getXMLpolygonannotations(self):
    return XMLPolygonAnnotationReader(self.annotationspolygonsxmlfile, self.pscale, self.apscale).getXMLpolygonannotations()

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
        if zoomlevel[0].imagewidth < 2000:
          break
      xresolution = zoomlevel.xresolution
      yresolution = zoomlevel.yresolution
      qpscale = zoomlevel.qpscale
      xposition = zoomlevel.xposition
      yposition = zoomlevel.yposition
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
          apscale=f.apscale,
          fname=self.jpgfilename.name,
          img="00001234",
          pscale=qpscale,
        )
      ]

      #build a jpg colored thumbnail from the qptiff, to illustrate the slide
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
  def apscale(self):
    return self.getqptiffcsv()[0].apscale

  @property
  def overlaps(self):
    return self.getoverlaps()

  def getconstants(self):
    pscales = {name: getattr(self, name) for name in ("pscale", "qpscale", "apscale")}
    constants = [
      Constant(
        name='fwidth',
        value=self.fwidth,
        unit='pixels',
        description='field width',
        **pscales,
      ),
      Constant(
        name='fheight',
        value=self.fheight,
        unit='pixels',
        description='field height',
        **pscales,
      ),
      Constant(
        name='flayers',
        value=self.flayers,
        unit='',
        description='field depth',
        **pscales,
      ),
      Constant(
        name='locx',
        value=self.samplelocation[0],
        unit='microns',
        description='xlocation',
        **pscales,
      ),
      Constant(
        name='locy',
        value=self.samplelocation[1],
        unit='microns',
        description='ylocation',
        **pscales,
      ),
      Constant(
        name='locz',
        value=self.samplelocation[2],
        unit='microns',
        description='zlocation',
        **pscales,
      ),
      Constant(
        name='xposition',
        value=units.convertpscale(self.xposition, self.qpscale, self.pscale),
        unit='microns',
        description='slide x offset',
        **pscales,
      ),
      Constant(
        name='yposition',
        value=units.convertpscale(self.yposition, self.qpscale, self.pscale),
        unit='microns',
        description='slide y offset',
        **pscales,
      ),
      Constant(
        name='qpscale',
        value=self.qpscale,
        unit='pixels/micron',
        description='scale of the QPTIFF image',
        **pscales,
      ),
      Constant(
        name='apscale',
        value=self.apscale,
        unit='pixels/micron',
        description='scale of the QPTIFF image used for annotation',
        **pscales,
      ),
      Constant(
        name='pscale',
        value=self.pscale,
        unit='pixels/micron',
        description='scale of the HPF images',
        **pscales,
      ),
      Constant(
        name='nclip',
        value=self.nclip * self.onepixel,
        unit='pixels',
        description='pixels to clip off the edge after warping',
        **pscales,
      ),
      Constant(
        name="resolutionbits",
        value=self.resolutionbits,
        unit="",
        description="number of significant bits in the im3 files",
        **pscales,
      ),
      Constant(
        name="gainfactor",
        value=self.gainfactor,
        unit="",
        description="the gain of the A/D amplifier for the im3 files",
        **pscales,
      ),
      Constant(
        name="binningx",
        value=self.camerabinningx,
        unit="pixels",
        description="the number of adjacent pixels coadded",
        **pscales,
      ),
      Constant(
        name="binningy",
        value=self.camerabinningy,
        unit="pixels",
        description="the number of adjacent pixels coadded",
        **pscales,
      ),
    ]
    return constants

class PrepDbSample(PrepDbSampleBase, DbloadSampleBase):
  def writebatch(self):
    self.logger.info("write batch")
    self.writecsv("batch", self.getbatch())

  def writerectangles(self):
    self.logger.info("write rectangles")
    self.writecsv("rect", self.rectangles)

  def writeglobals(self):
    if not self.globals: return
    self.logger.info("write globals")
    self.writecsv("globals", self.globals)

  def writeannotations(self):
    self.logger.info("write annotations")
    self.writecsv("annotations", self.annotations, rowclass=Annotation)

  def writeexposures(self):
    self.logger.info("write exposure times")
    self.writecsv("exposures", self.exposuretimes, rowclass=ExposureTime)

  def writeregions(self):
    self.logger.info("write regions")
    self.writecsv("regions", self.regions, rowclass=Region)

  def writevertices(self):
    self.logger.info("write vertices")
    self.writecsv("vertices", self.vertices, rowclass=Vertex)

  def writeqptiffcsv(self):
    self.logger.info("write qptiff csv")
    self.writecsv("qptiff", self.getqptiffcsv())

  def writeqptiffjpg(self):
    self.logger.info("write qptiff jpg")
    img = self.getqptiffimage()
    img.save(self.jpgfilename)

  def writeoverlaps(self):
    self.logger.info("write overlaps")
    self.writecsv("overlap", self.overlaps)

  def writeconstants(self):
    self.logger.info("write constants")
    self.writecsv("constants", self.getconstants())

  def writemetadata(self):
    self.dbload.mkdir(parents=True, exist_ok=True)
    self.writerectangles()
    self.writeexposures()
    self.writeoverlaps()
    self.writeannotations()
    self.writebatch()
    self.writeconstants()
    self.writeglobals()
    self.writeqptiffcsv()
    self.writeqptiffjpg()
    self.writevertices()
    self.writeregions()

  @property
  def inputfiles(self):
    return [
      self.annotationspolygonsxmlfile,
      self.annotationsxmlfile,
      self.fullxmlfile,
      self.parametersxmlfile,
      self.qptifffilename,
      self.scanfolder/"MSI",
    ]

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, **otherrootkwargs):
    dbload = dbloadroot/SlideID/"dbload"
    return [
      dbload/f"{SlideID}_annotations.csv",
      dbload/f"{SlideID}_batch.csv",
      dbload/f"{SlideID}_constants.csv",
      dbload/f"{SlideID}_exposures.csv",
      dbload/f"{SlideID}_overlap.csv",
      dbload/f"{SlideID}_qptiff.csv",
      dbload/f"{SlideID}_qptiff.jpg",
      dbload/f"{SlideID}_rect.csv",
      dbload/f"{SlideID}_regions.csv",
      dbload/f"{SlideID}_vertices.csv",
    ]

  @classmethod
  def workflowdependencies(cls):
    return [ShredXML] + super().workflowdependencies()

  @classmethod
  def logstartregex(cls):
    old = super().logstartregex()
    new = "prepSample started"
    return rf"(?:{old}|{new})"

  @classmethod
  def logendregex(cls):
    old = super().logendregex()
    new = "prepSample end"
    return rf"(?:{old}|{new})"

def main(args=None):
  p = argparse.ArgumentParser()
  p.add_argument("root")
  p.add_argument("samp")
  p.add_argument("--units", type=units.setup)
  p.add_argument("--dbload-root")
  args = p.parse_args(args=args)
  kwargs = {"root": args.root, "samp": args.samp, "dbloadroot": args.dbload_root}
  s = PrepDbSample(**kwargs)
  s.writemetadata()

if __name__ == "__main__":
  main()
