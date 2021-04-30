import more_itertools, os, pathlib

from astropath.slides.csvscan.csvscancohort import CsvScanCohort
from astropath.slides.csvscan.csvscansample import LoadFile, CsvScanSample

from .testbase import assertAlmostEqual, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestCsvScan(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    testroot = thisfolder/"csvscan_test_for_jenkins"
    yield thisfolder/"data"/"sampledef.csv", testroot

    for SlideID in "M206",:
      newdbload = testroot/SlideID/"dbload"
      newdbload.mkdir(parents=True, exist_ok=True)
      newtables = testroot/SlideID/"inform_data"/"Phenotyped"/"Results"/"Tables"
      newtables.mkdir(parents=True, exist_ok=True)

      for olddbload in thisfolder/"data"/SlideID/"dbload", thisfolder/"reference"/"geom"/SlideID, thisfolder/"reference"/"annowarp"/SlideID:
        for csv in olddbload.glob("*.csv"):
          if csv == thisfolder/"data"/SlideID/"dbload"/f"{SlideID}_vertices.csv": continue
          if csv == thisfolder/"data"/SlideID/"dbload"/f"{SlideID}_regions.csv": continue
          yield csv, newdbload

      oldtables = thisfolder/"data"/SlideID/"inform_data"/"Phenotyped"/"Results"/"Tables"
      for csv in oldtables.glob("*.csv"):
        yield csv, newtables
      

  @property
  def outputfilenames(self):
    return [
      thisfolder/"csvscan_test_for_jenkins"/SlideID/f"{SlideID}_{csv}.csv"
      for csv in ("loadfiles",)
      for SlideID in ("M206",)
    ]

  def testCsvScan(self, SlideID="M206", units="safe", selectrectangles=[1]):
    root = thisfolder/"csvscan_test_for_jenkins"
    geomroot = thisfolder/"reference"/"geomcell"
    args = [os.fspath(root), "--geomroot", os.fspath(geomroot), "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits"]
    if selectrectangles is not None:
      args.append("--selectrectangles")
      for rid in selectrectangles: args.append(str(rid))
    CsvScanCohort.runfromargumentparser(args=args)

    s = CsvScanSample(root=root, geomroot=geomroot, samp=SlideID)
    filename = s.csv("loadfiles")
    reffolder = thisfolder/"reference"/"csvscan"/SlideID
    reference = reffolder/filename.name

    try:
      rows = s.readtable(filename, LoadFile, header=False)
      targetrows = s.readtable(reference, LoadFile, header=False)
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testCsvScanFastUnits(self, SlideID="M206", **kwargs):
    self.testCsvScan(SlideID=SlideID, units="fast", **kwargs)
