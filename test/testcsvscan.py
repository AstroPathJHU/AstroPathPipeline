import more_itertools, os, pathlib

from astropath.slides.csvscan.csvscancohort import CsvScanCohort
from astropath.slides.csvscan.csvscansample import LoadFile, CsvScanSample

from .testbase import assertAlmostEqual, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestCsvScan(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    for SlideID in "M206",:
      newdbload = thisfolder/"csvscan_test_for_jenkins"/SlideID/"dbload"

      for olddbload in thisfolder/"data"/SlideID/"dbload",:
        for csv in olddbload.glob("*.csv"):
          yield csv, newdbload

  @property
  def outputfilenames(self):
    return [
      thisfolder/"csvscan_test_for_jenkins"/SlideID/f"{SlideID}_{csv}.csv"
      for csv in ("loadfiles",)
      for SlideID in ("M206",)
    ]

  def testCsvScan(self, SlideID="M206", units="safe", selectrectangles=None):
    root = thisfolder/"data"
    dbloadroot = thisfolder/"csvscan_test_for_jenkins"
    args = [os.fspath(root), "--dbloadroot", os.fspath(dbloadroot), "--geomroot", os.fspath(dbloadroot), "--phenotyperoot", os.fspath(dbloadroot), "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits"]
    if selectrectangles is not None:
      args.append("--selectrectangles")
      for rid in selectrectangles: args.append(str(rid))
    CsvScanCohort.runfromargumentparser(args=args)

    s = CsvScanSample(root=thisfolder/"data", dbloadroot=dbloadroot, samp=SlideID)
    filename = s.csv("loadfiles")
    reffolder = thisfolder/"reference"/"csvscan"/SlideID
    reference = reffolder/filename.name

    try:
      rows = s.readtable(filename, LoadFile)
      targetrows = s.readtable(reference, LoadFile)
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testCsvScanFastUnits(self, SlideID="M206", **kwargs):
    self.testCsvScan(SlideID=SlideID, units="fast", **kwargs)
