import argparse, methodtools, numpy as np, PIL, skimage
from ...shared.argumentparser import DbloadArgumentParser
from ...shared.csvclasses import Constant, Batch, ExposureTime, QPTiffCsv
from ...shared.overlap import RectangleOverlapCollection
from ...shared.qptiff import QPTiff
from ...shared.sample import DbloadSampleBase, WorkflowSample, XMLLayoutReader
from ...utilities import units
from ...utilities.config import CONST as UNIV_CONST

class FixFW01SampleBase(ReadRectanglesIm3Base, WorkflowSample):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, readlayerfile=False, **kwargs)

  @classmethod
  def logmodule(self): return "fixfw01"

  def fixfw01(self, check=True):
    n = len(self.rectangles)
    for i, r in enumerate(self.rectangles):
      self.logger.debug(f"rectangle {i}/{n}")
      kwargs = {field: getattr(samp, field) for field in set(dataclassy.fields(type(r)))}
      assert kwargs["readlayerfile"]
      kwargs["readlayerfile"] = False
      newr = type(r)(**kwargs)
      assert newr.im3file != r.im3file

      newfile = newr.im3file
      with contextlib.ExitStack() as stack:
        oldim3 = stack.enter_context(r.using_im3())
        newim3 = stack.enter_context(newr.using_im3())
        
      if newfile.exists() and newfile.stat().st_size == 0:
        self.logger.warning(f"{newfile.name} is corrupt, removing it")
        newfile.unlink()
      if newfile.exists(): continue

      with r.using_im3() as im:
        

  def run(self, **kwargs):
    self.fixfw01(**kwargs)

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

  def run(self, *args, **kwargs): return self.writemetadata(*args, **kwargs)

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
