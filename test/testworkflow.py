import os, pathlib

from astropath.utilities.miscfileio import checkwindowsnewlines
from astropath.shared.workflow import Workflow

from .testbase import TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestWorkflow(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    testroot = thisfolder/"test_for_jenkins"/"workflow"/"Clinical_Specimen_0"
    dataroot = thisfolder/"data"
    for foldername in "Batch", "Clinical", "Ctrl", pathlib.Path("Control_TMA_1372_111_06.19.2019")/"dbload":
      old = dataroot/foldername
      new = testroot/foldername
      for csv in old.glob("*.csv"):
        if csv.name in ("BatchID_24.csv", "MergeConfig_24.csv"): continue
        yield csv, new

  @property
  def outputfilenames(self):
    folder = thisfolder/"test_for_jenkins"/"workflow"
    if not folder.exists(): return []
    copiedfiles = {copytofolder/copyfrom.name for copyfrom, copytofolder in self.filestocopy()}
    return [_ for _ in folder.rglob("*") if _.is_file() and _ not in copiedfiles]

  def setUp(self):
    super().setUp()
    slideids = "M21_1",

    testroot = thisfolder/"test_for_jenkins"/"workflow"/"Clinical_Specimen_0"
    dataroot = thisfolder/"data"

    with open(dataroot/"sampledef.csv") as f, open(testroot/"sampledef.csv", "w") as newf:
      for line in f:
        if line.strip() and line.split(",")[1] in ("SlideID",) + slideids:
          newf.write(line)

  def testWorkflowFastUnits(self, units="fast"):
    testfolder = thisfolder/"test_for_jenkins"/"workflow"
    root = testfolder/"Clinical_Specimen_0"
    datafolder = thisfolder/"data"
    shardedim3root = datafolder/"flatw"
    zoomroot = testfolder/"zoom"
    deepzoomroot = testfolder/"deepzoom"
    SlideID = "M21_1"
    (root/SlideID).mkdir(parents=True, exist_ok=True)
    selectrectangles = 1, 17, 18, 23, 40
    args = [os.fspath(root), "--shardedim3root", os.fspath(shardedim3root), "--im3root", os.fspath(datafolder), "--informdataroot", os.fspath(datafolder), "--zoomroot", os.fspath(zoomroot), "--deepzoomroot", os.fspath(deepzoomroot), "--selectrectangles", *(str(_) for _ in selectrectangles), "--layers", "1", "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits", "--njobs", "3"]
    try:
      Workflow.runfromargumentparser(args=args)
      assert (root/"dbload"/"project0_loadfiles.csv").exists()
      for filename in testfolder.rglob("*.log"):
        checkwindowsnewlines(filename)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()
