import os, pathlib

from astropath.shared.workflow import Workflow

from .testbase import TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestWorkflow(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    for foldername in "Batch", "Clinical", "Ctrl", pathlib.Path("Control_TMA_1372_111_06.19.2019")/"dbload":
      old = dataroot/foldername
      new = testroot/foldername
      for csv in old.glob("*.csv"):
        yield csv, new

  @property
  def outputfilenames(self):
    folder = thisfolder/"workflow_test_for_jenkins"
    if not folder.exists(): return []
    copiedfiles = {copytofolder/copyfrom.name for copyfrom, copytofolder in self.filestocopy()}
    return [_ for _ in folder.rglob("*") if _.is_file() and _ not in copiedfiles]

  def setUp(self):
    super().setUp()
    slideids = "M206",

    testroot = thisfolder/"csvscan_test_for_jenkins"/"Clinical_Specimen_0"
    dataroot = thisfolder/"data"

    with open(dataroot/"sampledef.csv") as f, open(testroot/"sampledef.csv", "w") as newf:
      for line in f:
        if line.strip() and line.split(",")[1] in ("SlideID",) + slideids:
          newf.write(line)

  def testWorkflowFastUnits(self, units="fast"):
    testfolder = thisfolder/"workflow_test_for_jenkins"
    root = testfolder/"Clinical_Specimen_0"
    datafolder = thisfolder/"data"
    root2 = datafolder/"flatw"
    zoomroot = testfolder/"zoom"
    deepzoomroot = testfolder/"deepzoom"
    SlideID = "M21_1"
    (root/SlideID).mkdir(parents=True, exist_ok=True)
    selectrectangles = 1, 17, 18, 23, 40
    args = [os.fspath(root), os.fspath(root2), "--im3root", os.fspath(datafolder), "--informdataroot", os.fspath(datafolder), "--zoomroot", os.fspath(zoomroot), "--deepzoomroot", os.fspath(deepzoomroot), "--selectrectangles", *(str(_) for _ in selectrectangles), "--layers", "1", "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits"]
    try:
      Workflow.runfromargumentparser(args=args)
    finally:
      self.removeoutput()
