import more_itertools, os, pathlib

from astropath.slides.csvscan.csvscancohort import CsvScanCohort
from astropath.slides.csvscan.csvscansample import LoadFile, CsvScanSample
from astropath.utilities.misc import commonroot

from .testbase import assertAlmostEqual, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestCsvScan(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    testroot = thisfolder/"csvscan_test_for_jenkins"
    dataroot = thisfolder/"data"
    yield dataroot/"sampledef.csv", testroot

    for foldername in "Batch", "Clinical", "Ctrl", pathlib.Path("Control_TMA_1372_111_06.19.2019")/"dbload":
      old = dataroot/foldername
      new = testroot/foldername
      for csv in old.glob("*.csv"):
        yield csv, new

    for SlideID in "M206",:
      newdbload = testroot/SlideID/"dbload"
      newtables = testroot/SlideID/"inform_data"/"Phenotyped"/"Results"/"Tables"

      for olddbload in dataroot/SlideID/"dbload", thisfolder/"reference"/"geom"/SlideID, thisfolder/"reference"/"annowarp"/SlideID:
        for csv in olddbload.glob("*.csv"):
          if csv == dataroot/SlideID/"dbload"/f"{SlideID}_vertices.csv": continue
          if csv == dataroot/SlideID/"dbload"/f"{SlideID}_regions.csv": continue
          yield csv, newdbload

      oldtables = dataroot/SlideID/"inform_data"/"Phenotyped"/"Results"/"Tables"
      for csv in oldtables.glob("*.csv"):
        yield csv, newtables

      yield dataroot/SlideID/"im3"/f"{SlideID}-mean.csv", testroot/SlideID/"im3"

  @property
  def outputfilenames(self):
    return [
      thisfolder/"csvscan_test_for_jenkins"/SlideID/f"{SlideID}_{csv}.csv"
      for csv in ("loadfiles",)
      for SlideID in ("M206",)
    ]

  def testCsvScan(self, SlideID="M206", units="safe", selectrectangles=[1], skipcheck=False):
    root = thisfolder/"csvscan_test_for_jenkins"
    geomroot = thisfolder/"reference"/"geomcell"
    args = [os.fspath(root), "--geomroot", os.fspath(geomroot), "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits"]
    if selectrectangles is not None:
      args.append("--selectrectangles")
      for rid in selectrectangles: args.append(str(rid))
    if skipcheck:
      args.append("--skip-check")
    CsvScanCohort.runfromargumentparser(args=args)

    s = CsvScanSample(root=root, geomroot=geomroot, samp=SlideID)
    filename = s.csv("loadfiles")
    reffolder = thisfolder/"reference"/"csvscan"/SlideID
    reference = reffolder/filename.name

    try:
      rows = s.readtable(filename, LoadFile, header=False)
      targetrows = s.readtable(reference, LoadFile, header=False)
      for row in rows:
        folder = s.mainfolder
        if row.tablename == "CellGeom":
          folder = s.geomroot/s.SlideID
        row.filename = row.filename.relative_to(folder).as_posix()
      targetcommonroot = commonroot(*(row.filename for row in targetrows))
      for row in targetrows:
        row.filename = row.filename.relative_to(targetcommonroot).as_posix()
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target)

      filename = s.root/"dbload"/"project0_loadfiles.csv"
      reference = thisfolder/"reference"/"csvscan"/filename.name
      rows = s.readtable(filename, LoadFile, header=False)
      targetrows = s.readtable(reference, LoadFile, header=False)
      for row in rows:
        folder = s.root
        if row.tablename == "CellGeom":
          folder = s.geomroot/s.SlideID
        row.filename = row.filename.relative_to(folder).as_posix()
      targetcommonroot = commonroot(*(row.filename for row in targetrows))
      for row in targetrows:
        row.filename = row.filename.relative_to(targetcommonroot).as_posix()
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testCsvScanFastUnits(self, **kwargs):
    self.testCsvScan(units="fast", **kwargs)

  def testCsvScanNoCheck(self, **kwargs):
    self.testCsvScan(skipcheck=True)
