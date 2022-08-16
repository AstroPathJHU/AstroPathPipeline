import contextlib, csv, itertools, job_lock, logging, more_itertools, os, pathlib, re, shutil
from astropath.shared.csvclasses import Batch, Constant, ExposureTime, QPTiffCsv, ROIGlobals
from astropath.shared.logging import getlogger
from astropath.shared.overlap import Overlap
from astropath.shared.rectangle import Rectangle
from astropath.shared.sample import SampleDef
from astropath.slides.prepdb.prepdbcohort import PrepDbCohort
from astropath.slides.prepdb.prepdbsample import PrepDbSample
from astropath.utilities.miscfileio import checkwindowsnewlines
from astropath.utilities.version.git import thisrepo
from .testbase import assertAlmostEqual, compare_two_images, temporarilyreplace, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestFixFW01(TestBaseCopyInput, TestBaseSaveOutput):
  testrequirecommit = thisrepo.getcommit("cf271f3a")

  @property
  def outputfilenames(self):

  @classmethod
  def filestocopy(cls):
    oldflatw = thisfolder/"data"/"flatw"
    newflatw = thisfolder/"test_for_jenkins"/"fixfw01"
    
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
    for filename in "M206_Scan1_annotations.xml", "M206_Scan1.qptiff":
      yield oldScan1/filename, newScan1
    oldMSI = oldScan1/"MSI"
    newMSI = newScan1/"MSI"
    for filename in oldMSI.glob("*.im3"):
      yield filename, newMSI

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

    skipqptiff = self.skipqptiff(SlideID)

    args = [os.fspath(thisfolder/"data"), "--sampleregex", SlideID, "--debug", "--units", units, "--xmlfolder", os.fspath(thisfolder/"data"/"raw"/SlideID), "--allow-local-edits", "--ignore-dependencies", "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(dbloadroot)] + moreargs
    if skipqptiff:
      args.append("--skip-qptiff")

    try:
      sample = PrepDbSample(thisfolder/"data", SlideID, uselogfiles=False, xmlfolders=[thisfolder/"data"/"raw"/SlideID], dbloadroot=dbloadroot, logroot=dbloadroot)
      PrepDbCohort.runfromargumentparser(args) #this should not run anything
      with open(sample.csv("rect")) as f: assert not f.read().strip()
      PrepDbCohort.runfromargumentparser(args + ["--require-commit", str(self.testrequirecommit.parents[0].parents[0])])

      rectangles = None
      for filename, cls, extrakwargs in (
        (f"{SlideID}_batch.csv", Batch, {}),
        (f"{SlideID}_constants.csv", Constant, {}),
        (f"{SlideID}_exposures.csv", ExposureTime, {}),
        (f"{SlideID}_globals.csv", ROIGlobals, {}),
        (f"{SlideID}_qptiff.csv", QPTiffCsv, {}),
        (f"{SlideID}_rect.csv", Rectangle, {}),
        (f"{SlideID}_overlap.csv", Overlap, {"nclip": sample.nclip}),
      ):
        if filename == "M21_1_globals.csv": continue
        if filename == "ZW2_exposures.csv": continue #that file is huge and unnecessary
        if skipqptiff and cls in (Constant, QPTiffCsv):
          self.assertFalse((dbloadroot/SlideID/"dbload"/filename).exists())
          continue
        if cls == Overlap:
          extrakwargs["rectangles"] = rectangles
        sample.logger.info(f"comparing {filename}")
        try:
          rows = sample.readtable(dbloadroot/SlideID/"dbload"/filename, cls, checkorder=True, checknewlines=True, extrakwargs=extrakwargs)
          targetrows = sample.readtable(thisfolder/"data"/"reference"/"prepdb"/SlideID/"dbload"/filename, cls, checkorder=True, checknewlines=True, extrakwargs=extrakwargs)
          for i, (row, target) in enumerate(more_itertools.zip_equal(rows, targetrows)):
            assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
          if cls == Rectangle:
            rectangles = rows
        except Exception as e:
          raise type(e)("Error in "+filename)

      if not skipqptiff:
        jpg = dbloadroot/SlideID/"dbload"/f"{SlideID}_qptiff.jpg"
        def refjpg(i): return thisfolder/"data"/"reference"/"prepdb"/SlideID/"dbload"/f"{SlideID}_qptiff_{i}.jpg"
        try:
          compare_two_images(jpg, refjpg(1))
        except AssertionError:
          try:
            compare_two_images(jpg, refjpg(2))
          except AssertionError:
            compare_two_images(jpg, refjpg(3))

      logs = (
        dbloadroot/"logfiles"/"prepdb.log",
        dbloadroot/SlideID/"logfiles"/f"{SlideID}-prepdb.log",
      )
      for log in logs:
        checkwindowsnewlines(log)
    except:
      if removeoutput:
        #don't save empty files that were created in setUp for testing the dependency check
        for folder in dbloadroot.iterdir():
          if folder.name != SlideID and folder.is_dir():
            shutil.rmtree(folder)
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
    self.testPrepDb(SlideID="M206", units="fast_microns", moreargs=["--im3root", os.fspath(testroot), "--xmlfolder", os.fspath(thisfolder/"data"/"M206"/"im3"/"xml")])

  def testPrepDbZW2(self):
    testroot = thisfolder/"test_for_jenkins"/"prepdb"
    sampledef = testroot/"sampledef.csv"
    with open(sampledef, newline="") as f:
      contents = f.read()
    newcontents = re.sub(r"(\s[0-9]+,ZW2.*)1(\s)", r"\g<1>0\g<2>", contents, re.MULTILINE)
    self.assertNotEqual(newcontents, contents)
    moreargs = ["--sampledefroot", os.fspath(testroot)]
    with temporarilyreplace(sampledef, newcontents):
      with self.assertRaises(EOFError):
        self.testPrepDb(SlideID="ZW2", units="fast_microns", moreargs=moreargs, removeoutput=False)
      self.testPrepDb(SlideID="ZW2", units="fast_microns", moreargs=moreargs + ["--include-bad-samples"])
