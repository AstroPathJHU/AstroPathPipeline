import contextlib, csv, itertools, job_lock, logging, more_itertools, numpy as np, os, pathlib, PIL.Image, re, sys
from astropath.shared.csvclasses import Annotation, Batch, Constant, ExposureTime, QPTiffCsv, Region, ROIGlobals, Vertex
from astropath.shared.logging import getlogger
from astropath.shared.overlap import Overlap
from astropath.shared.rectangle import Rectangle
from astropath.shared.sample import SampleDef
from astropath.slides.prepdb.prepdbcohort import PrepDbCohort
from astropath.slides.prepdb.prepdbsample import PrepDbSample
from astropath.utilities.miscfileio import checkwindowsnewlines
from astropath.utilities.version.git import thisrepo
from .testbase import assertAlmostEqual, temporarilyreplace, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestPrepDb(TestBaseCopyInput, TestBaseSaveOutput):
  testrequirecommit = thisrepo.getcommit("cf271f3a")

  @property
  def outputfilenames(self):
    SlideIDs = "M21_1", "YZ71", "M206", "ZW2"
    return [
      thisfolder/"test_for_jenkins"/"prepdb"/SlideID/"dbload"/filename.name
      for SlideID in SlideIDs
      for ext in ("csv", "jpg")
      for filename in (thisfolder/"data"/"reference"/"prepdb"/SlideID).glob(f"*.{ext}")
    ] + [ #this file is not compared to reference because it's huge
      thisfolder/"test_for_jenkins"/"prepdb"/"ZW2"/"dbload"/"ZW2_exposures.csv",
    ] + [
      thisfolder/"test_for_jenkins"/"prepdb"/SlideID/"logfiles"/f"{SlideID}-prepdb.log"
      for SlideID in SlideIDs
    ] + [
      thisfolder/"test_for_jenkins"/"prepdb"/SlideID/"dbload"/f"{SlideID}_qptiff.jpg"
      for SlideID in SlideIDs
    ] + [
      thisfolder/"test_for_jenkins"/"prepdb"/"logfiles"/"prepdb.log",
    ]

  def setUp(self):
    stack = self.__stack = contextlib.ExitStack()
    super().setUp()
    try:
      slideids = "M21_1", "M206", "YZ71", "ZW2"
      testroot = thisfolder/"test_for_jenkins"/"prepdb"
      for SlideID in slideids:
        logfolder = testroot/SlideID/"logfiles"
        logfolder.mkdir(exist_ok=True, parents=True)

        filename = logfolder/f"{SlideID}-prepdb.log"
        assert stack.enter_context(job_lock.JobLock(filename))
        filename.unlink()
        with getlogger(root=testroot, samp=SampleDef(SlideID=SlideID, Project=0, Cohort=0), module="prepdb", reraiseexceptions=False, uselogfiles=True, printthreshold=logging.CRITICAL+1) as logger:
          logger.info("testing")
        with open(filename, newline="") as f:
          f, f2 = itertools.tee(f)
          startregex = re.compile(PrepDbSample.logstartregex())
          reader = csv.DictReader(f, fieldnames=("Project", "Cohort", "SlideID", "message", "time"), delimiter=";")
          for row in reader:
            match = startregex.match(row["message"])
            istag = not bool(match.group("commit"))
            if match: break
          else:
            assert False
          contents = "".join(f2)

        usecommit = self.testrequirecommit.parents[0]
        if istag:
          contents = contents.replace(match.group("version"), f"{match.group('version')}.dev0+g{usecommit.shorthash(8)}")
        else:
          contents = contents.replace(match.group("commit"), usecommit.shorthash(8))

        with open(filename, "w", newline="") as f:
          f.write(contents)

        dbloadfolder = testroot/SlideID/"dbload"
        dbloadfolder.mkdir(exist_ok=True, parents=True)
        for filename in "batch.csv", "exposures.csv", "overlap.csv", "rect.csv", "constants.csv", "qptiff.csv", "qptiff.jpg", "annotations.csv", "regions.csv", "vertices.csv", "globals.csv":
          if self.skipannotations(SlideID) and filename in ("regions.csv", "annotations.csv", "vertices.csv"): continue
          if self.skipqptiff(SlideID) and filename in ("constants.csv", "qptiff.csv", "qptiff.jpg"): continue
          if SlideID == "M21_1" and filename == "globals.csv": continue
          (dbloadfolder/f"{SlideID}_{filename}").touch()
    except:
      stack.close()
      raise

  def tearDown(self):
    super().tearDown()
    self.__stack.close()

  @classmethod
  def setUpClass(cls):
    from .data.M206.im3.Scan1.assembleqptiff import assembleqptiff
    assembleqptiff()
    super().setUpClass()


  @classmethod
  def filestocopy(cls):
    oldroot = thisfolder/"data"
    newroot = thisfolder/"test_for_jenkins"/"prepdb"
    for filename in "sampledef.csv",:
      yield oldroot/filename, newroot
    oldfolder = oldroot/"M206"
    newfolder = newroot/"M206"
    for filename in ():
      yield oldfolder/filename, newfolder
    oldim3 = oldfolder/"im3"
    newim3 = newfolder/"im3"
    for filename in ():
      yield oldim3/filename, newim3
    oldScan1 = oldim3/"Scan1"
    newScan1 = newim3/"Scan1"
    for filename in "M206_Scan1.annotations.polygons.xml", "M206_Scan1_annotations.xml", "M206_Scan1.qptiff":
      yield oldScan1/filename, newScan1
    oldMSI = oldScan1/"MSI"
    newMSI = newScan1/"MSI"
    for filename in oldMSI.glob("*.im3"):
      yield filename, newMSI

  @classmethod
  def skipannotations(cls, SlideID):
    return {
      "M206": False,
      "M21_1": False,
      "YZ71": True,
      "ZW2": False
    }[SlideID]
  @classmethod
  def skipqptiff(cls, SlideID):
    return {
      "M206": False,
      "M21_1": False,
      "YZ71": False,
      "ZW2": True,
    }[SlideID]

  def testPrepDb(self, SlideID="M21_1", units="safe", moreargs=[], removeoutput=True):
    dbloadroot = thisfolder/"test_for_jenkins"/"prepdb"

    skipannotations = self.skipannotations(SlideID)
    skipqptiff = self.skipqptiff(SlideID)

    args = [os.fspath(thisfolder/"data"), "--sampleregex", SlideID, "--debug", "--units", units, "--xmlfolder", os.fspath(thisfolder/"data"/"raw"/SlideID), "--allow-local-edits", "--ignore-dependencies", "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(dbloadroot)] + moreargs
    if skipannotations:
      args.append("--skip-annotations")
    if skipqptiff:
      args.append("--skip-qptiff")

    try:
      sample = PrepDbSample(thisfolder/"data", SlideID, uselogfiles=False, xmlfolders=[thisfolder/"data"/"raw"/SlideID], dbloadroot=dbloadroot, logroot=dbloadroot)
      PrepDbCohort.runfromargumentparser(args) #this should not run anything
      PrepDbCohort.runfromargumentparser(args + ["--require-commit", str(self.testrequirecommit.parents[0])]) #this should not run anything either
      with open(sample.csv("rect")) as f: assert not f.read().strip()
      PrepDbCohort.runfromargumentparser(args + ["--require-commit", str(self.testrequirecommit)])

      rectangles = None
      for filename, cls, extrakwargs in (
        (f"{SlideID}_annotations.csv", Annotation, {}),
        (f"{SlideID}_batch.csv", Batch, {}),
        (f"{SlideID}_constants.csv", Constant, {}),
        (f"{SlideID}_exposures.csv", ExposureTime, {}),
        (f"{SlideID}_globals.csv", ROIGlobals, {}),
        (f"{SlideID}_qptiff.csv", QPTiffCsv, {}),
        (f"{SlideID}_rect.csv", Rectangle, {}),
        (f"{SlideID}_overlap.csv", Overlap, {"nclip": sample.nclip}),
        (f"{SlideID}_vertices.csv", Vertex, {}),
        (f"{SlideID}_regions.csv", Region, {}),
      ):
        if filename == "M21_1_globals.csv": continue
        if filename == "ZW2_exposures.csv": continue #that file is huge and unnecessary
        if skipannotations and cls in (Annotation, Vertex, Region):
          self.assertFalse((dbloadroot/SlideID/"dbload"/filename).exists())
          continue
        if skipqptiff and cls in (Constant, QPTiffCsv):
          self.assertFalse((dbloadroot/SlideID/"dbload"/filename).exists())
          continue
        if cls == Overlap:
          extrakwargs["rectangles"] = rectangles
        sample.logger.info(f"comparing {filename}")
        try:
          rows = sample.readtable(dbloadroot/SlideID/"dbload"/filename, cls, checkorder=True, checknewlines=True, extrakwargs=extrakwargs)
          targetrows = sample.readtable(thisfolder/"data"/"reference"/"prepdb"/SlideID/filename, cls, checkorder=True, checknewlines=True, extrakwargs=extrakwargs)
          for i, (row, target) in enumerate(more_itertools.zip_equal(rows, targetrows)):
            assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
          if cls == Rectangle:
            rectangles = rows
        except Exception as e:
          raise type(e)("Error in "+filename)

      platform = sys.platform
      if platform == "darwin": platform = "linux"
      if not skipqptiff and platform != "win32":
        with PIL.Image.open(dbloadroot/SlideID/"dbload"/f"{SlideID}_qptiff.jpg") as img, \
             PIL.Image.open(thisfolder/"data"/"reference"/"prepdb"/SlideID/f"{SlideID}_qptiff_{platform}.jpg") as targetimg:
          np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))

      logs = (
        dbloadroot/"logfiles"/"prepdb.log",
        dbloadroot/SlideID/"logfiles"/f"{SlideID}-prepdb.log",
      )
      for log in logs:
        checkwindowsnewlines(log)
    except:
      if removeoutput:
        self.saveoutput()
      raise
    finally:
      if removeoutput:
        self.removeoutput()

  def testPrepDbFastUnits(self, SlideID="M21_1"):
    self.testPrepDb(SlideID, units="fast")

  def testPrepDbPolaris(self, **kwargs):
    from .data.YZ71.im3.Scan3.assembleqptiff import assembleqptiff
    assembleqptiff()
    self.testPrepDb(SlideID="YZ71", **kwargs)
  def testPrepDbPolarisFastUnits(self):
    self.testPrepDbPolaris(units="fast")

  def testPrepDbM206FastUnits(self):
    testroot = thisfolder/"test_for_jenkins"/"prepdb"
    xmlfile = testroot/"M206"/"im3"/"Scan1"/"M206_Scan1.annotations.polygons.xml"
    with open(xmlfile) as f:
      contents = f.read()
    with temporarilyreplace(xmlfile, contents.replace("Tumor", "Good Tissue").replace("Good tissue", "Tumor")):
      self.testPrepDb(SlideID="M206", units="fast_microns", moreargs=["--rename-annotation", "Good tissue", "Tumor", "--rename-annotation", "Tumor", "Good Tissue", "--im3root", os.fspath(testroot), "--xmlfolder", os.fspath(thisfolder/"data"/"M206"/"im3"/"xml")])

  def testPrepDBZW2(self):
    testroot = thisfolder/"test_for_jenkins"/"prepdb"
    sampledef = testroot/"sampledef.csv"
    with open(sampledef, newline="") as f:
      contents = f.read()
    newcontents = re.sub(r"(\s[0-9]+,ZW2.*)1(\s)", r"\g<1>0\g<2>", contents, re.MULTILINE)
    self.assertNotEqual(newcontents, contents)
    moreargs = ["--sampledefroot", os.fspath(testroot)]
    with temporarilyreplace(sampledef, newcontents):
      with self.assertRaises(TypeError):
        self.testPrepDb(SlideID="ZW2", units="fast_microns", moreargs=moreargs, removeoutput=False)
      self.testPrepDb(SlideID="ZW2", units="fast_microns", moreargs=moreargs + ["--include-bad-samples"])
