import argparse, methodtools, numpy as np, PIL, skimage
from ...shared.argumentparser import DbloadArgumentParser, XMLPolygonReaderArgumentParser
from ...shared.csvclasses import Annotation, Constant, Batch, ExposureTime, QPTiffCsv, Region, Vertex
from ...shared.overlap import RectangleOverlapCollection
from ...shared.qptiff import QPTiff
from ...shared.sample import DbloadSampleBase, WorkflowSample, XMLLayoutReader, XMLPolygonAnnotationReaderSample
from ...utilities import units
from ...utilities.config import CONST as UNIV_CONST

class PrepDbArgumentParser(DbloadArgumentParser, XMLPolygonReaderArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--skip-annotations", action="store_true", help="do not check the annotations for validity and do not write the annotations, vertices, and regions csvs (they will be written later, in the annowarp step)")
    p.add_argument("--skip-qptiff", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--margin", type=int, help="minimum number of pixels between the tissue and the wsi edge", default=1024)
    return p

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "skipannotations": parsed_args_dict.pop("skip_annotations"),
      "_skipqptiff": parsed_args_dict.pop("skip_qptiff"),
    }

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "margin": parsed_args_dict.pop("margin"),
    }

class PrepDbSampleBase(XMLLayoutReader, DbloadSampleBase, XMLPolygonAnnotationReaderSample, RectangleOverlapCollection, WorkflowSample, units.ThingWithQpscale, units.ThingWithApscale):
  """
  The prepdb stage of the pipeline extracts metadata for a sample from the `.xml` files
  and writes it out to `.csv` files.
  For more information, see README.md in this folder.
  """

  def __init__(self, *args, nclip=8, margin=1024, **kwargs):
    super().__init__(*args, **kwargs)
    self.__margin = margin
    self.__nclip = nclip

  @classmethod
  def logmodule(self): return "prepdb"

  @property
  def nclip(self): return self.__nclip * self.onepixel
  @property
  def margin(self): return self.__margin * self.onepixel

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

  @property
  def annotations(self): return self.getXMLpolygonannotations()[0]
  @property
  def regions(self): return self.getXMLpolygonannotations()[1]
  @property
  def vertices(self): return self.getXMLpolygonannotations()[2]

  @property
  def jpgfilename(self): return self.dbload/(self.SlideID+UNIV_CONST.QPTIFF_SUFFIX)

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
    if not self.qptifffilename.exists(): return 1
    return self.getqptiffcsv()[0].qpscale
  @property
  def apscale(self):
    if not self.qptifffilename.exists(): return 1
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
        value=self.nclip,
        unit='pixels',
        description='pixels to clip off the edge after warping',
        **pscales,
      ),
      Constant(
        name='margin',
        value=self.margin,
        unit='pixels',
        description='minimum margin between the tissue and the wsi edge',
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

class PrepDbSample(PrepDbSampleBase, PrepDbArgumentParser):
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

  def writemetadata(self, *, skipannotations=False, _skipqptiff=False):
    self.dbload.mkdir(parents=True, exist_ok=True)
    self.writerectangles()
    self.writeexposures()
    self.writeoverlaps()
    self.writebatch()
    self.writeglobals()
    if _skipqptiff:
      self.logger.warningglobal("as requested, not writing the qptiff info.  subsequent steps that rely on constants.csv may not work.")
    else:
      self.writeconstants()
      self.writeqptiffcsv()
      self.writeqptiffjpg()
    if skipannotations:
      self.logger.warningglobal("as requested, not checking annotations. the csvs will be written in the annowarp step.")
    else:
      self.writeannotations()
      self.writevertices()
      self.writeregions()

  run = writemetadata

  def inputfiles(self, *, skipannotations=False, _skipqptiff=False, **kwargs):
    result = super().inputfiles(**kwargs) + [
      self.annotationsxmlfile,
      self.fullxmlfile,
      self.parametersxmlfile,
    ]
    imagefolder = self.scanfolder/"MSI"
    if not imagefolder.exists():
      imagefolder = self.scanfolder/"flatw"
    result += [
      imagefolder,
    ]
    if not _skipqptiff:
      result += [
        self.qptifffilename,
      ]
    if not skipannotations:
      result += [
        self.annotationspolygonsxmlfile,
      ]
    return result

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, skipannotations=False, _skipqptiff=False, **otherkwargs):
    dbload = dbloadroot/SlideID/UNIV_CONST.DBLOAD_DIR_NAME
    return [
      dbload/f"{SlideID}_batch.csv",
      dbload/f"{SlideID}_exposures.csv",
      dbload/f"{SlideID}_overlap.csv",
      dbload/f"{SlideID}_rect.csv",
    ] + ([
      dbload/f"{SlideID}_constants.csv",
      dbload/f"{SlideID}_qptiff.csv",
      dbload/f"{SlideID}{UNIV_CONST.QPTIFF_SUFFIX}",
    ] if not _skipqptiff else []) + ([
      dbload/f"{SlideID}_annotations.csv",
      dbload/f"{SlideID}_regions.csv",
      dbload/f"{SlideID}_vertices.csv",
    ] if not skipannotations else [])

  @classmethod
  def workflowdependencyclasses(cls):
    return super().workflowdependencyclasses()

  @classmethod
  def logstartregex(cls):
    new = super().logstartregex()
    old = "prepSample started"
    return rf"(?:{old}|{new})"

  @classmethod
  def logendregex(cls):
    new = super().logendregex()
    old = "prepSample end"
    return rf"(?:{old}|{new})"

def main(args=None):
  PrepDbSample.runfromargumentparser(args)

if __name__ == "__main__":
  main()
