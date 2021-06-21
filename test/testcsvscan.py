import more_itertools, os, pathlib

from astropath.slides.csvscan.csvscancohort import CsvScanCohort
from astropath.slides.csvscan.csvscansample import LoadFile, CsvScanSample
from astropath.utilities.misc import commonroot

from .testbase import assertAlmostEqual, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestCsvScan(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    testroot = thisfolder/"test_for_jenkins"/"csvscan"/"Clinical_Specimen_0"
    dataroot = thisfolder/"data"

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
      thisfolder/"test_for_jenkins"/"csvscan"/"Clinical_Specimen_0"/SlideID/"dbload"/f"{SlideID}_{csv}.csv"
      for csv in ("loadfiles",)
      for SlideID in ("M206",)
    ] + [
      thisfolder/"test_for_jenkins"/"csvscan"/"Clinical_Specimen_0"/SlideID/"logfiles"/f"{SlideID}-csvscan.log"
      for SlideID in ("M206",)
    ] + [
      thisfolder/"test_for_jenkins"/"csvscan"/"Clinical_Specimen_0"/"dbload"/"project0_loadfiles.csv"
      thisfolder/"test_for_jenkins"/"csvscan"/"Clinical_Specimen_0"/"logfiles"/"csvscan.log"
    ]

  def setUp(self):
    super().setUp()
    stack = self.__stack = contextlib.ExitStack()
    slideids = "M206",

    testroot = thisfolder/"test_for_jenkins"/"csvscan"/"Clinical_Specimen_0"
    dataroot = thisfolder/"data"
    for SlideID in slideids:
      logfolder = testroot/SlideID/"logfiles"
      logfolder.mkdir(exist_ok=True, parents=True)
      from astropath.utilities.version import astropathversion
      for module in "annowarp", "geom", "geomcell":
        filename = logfolder/f"{SlideID}-{module}.log"
        assert stack.enter_context(job_lock.JobLock(filename))
        with open(filename, "w") as f:
          f.write(f"0;0;{SlideID};{module} {astropathversion};1900-01-01 00:00:00\n")
          f.write(f"0;0;{SlideID};end {module};1900-01-01 00:00:00\n")

    sampledef = testroot/"sampledef.csv"
    assert stack.enter_context(job_lock.JobLock(sampledef))
    with open(dataroot/"sampledef.csv") as f, open(sampledef, "w") as newf:
      for line in f:
        if line.strip() and line.split(",")[1] in ("SlideID",) + slideids:
          newf.write(line)

  def tearDown(self):
    super().tearDown()
    self.__stack.close()

  def testCsvScan(self, SlideID="M206", units="safe", selectrectangles=[1], skipcheck=False):
    root = thisfolder/"test_for_jenkins"/"csvscan"/"Clinical_Specimen_0"
    geomroot = thisfolder/"reference"/"geomcell"
    args = [os.fspath(root), "--geomroot", os.fspath(geomroot), "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits", "--rerun-finished"]
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
      rows = s.readtable(filename, LoadFile, header=False, checkorder=True, checknewlines=True)
      targetrows = s.readtable(reference, LoadFile, header=False, checkorder=True, checknewlines=True)
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
      rows = s.readtable(filename, LoadFile, header=False, checkorder=True, checknewlines=True)
      targetrows = s.readtable(reference, LoadFile, header=False, checkorder=True, checknewlines=True)
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
