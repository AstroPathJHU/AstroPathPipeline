import argparse, methodtools, numpy as np, PIL, skimage
from ...shared.argumentparser import DbloadArgumentParser
from ...shared.csvclasses import Constant, Batch, ExposureTime, QPTiffCsv
from ...shared.overlap import RectangleOverlapCollection
from ...shared.qptiff import QPTiff
from ...shared.sample import DbloadSampleBase, WorkflowSample, XMLLayoutReader
from ...utilities import units
from ...utilities.config import CONST as UNIV_CONST

class PrepDbArgumentParser(DbloadArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--skip-qptiff", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--margin", type=int, help="minimum number of pixels between the tissue and the wsi edge", default=1024)
    return p

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "_skipqptiff": parsed_args_dict.pop("skip_qptiff"),
    }

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "margin": parsed_args_dict.pop("margin"),
    }

class PrepDbSampleBase(XMLLayoutReader, DbloadSampleBase, RectangleOverlapCollection, WorkflowSample, units.ThingWithQpscale, units.ThingWithApscale):
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

      with zoomlevel.using_image(layers=(1, 2, 3, 4, 5)) as img:
        finalimg = np.tensordot(img, mix, axes=[2, 1])
      np.testing.assert_array_equal(finalimg.shape, shape)
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
        **pscales,
      ),
      Constant(
        name='fheight',
        value=self.fheight,
        **pscales,
      ),
      Constant(
        name='flayers',
        value=self.flayers,
        **pscales,
      ),
      Constant(
        name='locx',
        value=self.samplelocation[0],
        **pscales,
      ),
      Constant(
        name='locy',
        value=self.samplelocation[1],
        **pscales,
      ),
      Constant(
        name='locz',
        value=self.samplelocation[2],
        **pscales,
      ),
      Constant(
        name='xposition',
        value=units.convertpscale(self.xposition, self.qpscale, self.pscale),
        **pscales,
      ),
      Constant(
        name='yposition',
        value=units.convertpscale(self.yposition, self.qpscale, self.pscale),
        **pscales,
      ),
      Constant(
        name='qpscale',
        value=self.qpscale,
        **pscales,
      ),
      Constant(
        name='apscale',
        value=self.apscale,
        **pscales,
      ),
      Constant(
        name='pscale',
        value=self.pscale,
        **pscales,
      ),
      Constant(
        name='nclip',
        value=self.nclip,
        **pscales,
      ),
      Constant(
        name='margin',
        value=self.margin,
        **pscales,
      ),
      Constant(
        name="resolutionbits",
        value=self.resolutionbits,
        **pscales,
      ),
      Constant(
        name="gainfactor",
        value=self.gainfactor,
        **pscales,
      ),
      Constant(
        name="binningx",
        value=self.camerabinningx,
        **pscales,
      ),
      Constant(
        name="binningy",
        value=self.camerabinningy,
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

  def writeexposures(self):
    self.logger.info("write exposure times")
    self.writecsv("exposures", self.exposuretimes, rowclass=ExposureTime)

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

  def writemetadata(self, *, _skipqptiff=False):
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

  run = writemetadata

  def inputfiles(self, *, _skipqptiff=False, **kwargs):
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
    return result

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, _skipqptiff=False, **otherkwargs):
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
    ] if not _skipqptiff else [])

  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return super().workflowdependencyclasses(**kwargs)

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
