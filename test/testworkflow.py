import more_itertools, os, pathlib

from astropath.shared.workflow import Workflow
from astropath.utilities.misc import commonroot

from .testbase import assertAlmostEqual, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestWorkflow(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    testroot = thisfolder/"workflow_test_for_jenkins"/"Clinical_Specimen_0"
    dataroot = thisfolder/"data"
    yield dataroot/"sampledef.csv", testroot

    for foldername in "Batch", "Clinical", "Ctrl", pathlib.Path("Control_TMA_1372_111_06.19.2019")/"dbload":
      old = dataroot/foldername
      new = testroot/foldername
      for csv in old.glob("*.csv"):
        yield csv, new

    for SlideID in "M21_1",:
      newdbload = testroot/SlideID/"dbload"
      newtables = testroot/SlideID/"inform_data"/"Phenotyped"/"Results"/"Tables"

      for olddbload in ():
        for csv in olddbload.glob("*.csv"):
          if csv == dataroot/SlideID/"dbload"/f"{SlideID}_vertices.csv": continue
          if csv == dataroot/SlideID/"dbload"/f"{SlideID}_regions.csv": continue
          yield csv, newdbload

  @property
  def outputfilenames(self):
    folder = thisfolder/"workflow_test_for_jenkins"
    if not folder.exists(): return []
    copiedfiles = {copytofolder/copyfrom.name for copyfrom, copytofolder in self.filestocopy()}
    return [_ for _ in folder.rglob("*") if _.is_file() and _ not in copiedfiles]

  def testWorkflow(self, units="safe"):
    testfolder = thisfolder/"workflow_test_for_jenkins"
    root = testfolder/"Clinical_Specimen_0"
    datafolder = thisfolder/"data"
    root2 = datafolder/"flatw"
    zoomroot = testfolder/"zoom"
    deepzoomroot = testfolder/"deepzoom"
    SlideID = "M21_1"
    selectrectangles = 1, 17, 18, 23, 40
    args = [os.fspath(root), os.fspath(root2), "--im3root", os.fspath(datafolder), "--informdataroot", os.fspath(datafolder), "--zoomroot", os.fspath(zoomroot), "--deepzoomroot", os.fspath(deepzoomroot), "--selectrectangles", *(str(_) for _ in selectrectangles), "--layers", "1", "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits"]
    try:
      Workflow.runfromargumentparser(args=args)
    finally:
      self.removeoutput()

  def testWorkflowFastUnits(self, **kwargs):
    self.testWorkflow(units="fast", **kwargs)
