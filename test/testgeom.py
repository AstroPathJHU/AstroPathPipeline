import contextlib, job_lock, logging, more_itertools, os, pathlib

from astropath.shared.logging import getlogger
from astropath.shared.sample import SampleDef
from astropath.slides.geom.geomcohort import GeomCohort
from astropath.slides.geom.geomsample import Boundary, GeomSample

from .testbase import assertAlmostEqual, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestGeom(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    for SlideID in "M206", "M148":
      olddbload = thisfolder/"data"/SlideID/"dbload"
      newdbload = thisfolder/"test_for_jenkins"/"geom"/SlideID/"dbload"
      for csv in (
        "constants",
        "fields",
      ):
        yield olddbload/f"{SlideID}_{csv}.csv", newdbload

  @property
  def outputfilenames(self):
    return [
      thisfolder/"test_for_jenkins"/"geom"/SlideID/"dbload"/f"{SlideID}_{csv}.csv"
      for csv in ("tumorGeometry", "fieldGeometry")
      for SlideID in ("M206", "M148")
    ] + [
      thisfolder/"test_for_jenkins"/"geom"/SlideID/"logfiles"/f"{SlideID}-{log}.log"
      for log in ("geom",)
      for SlideID in ("M206", "M148")
    ] + [
      thisfolder/"test_for_jenkins"/"geom"/"logfiles"/f"{log}.log"
      for log in ("geom",)
    ]

  def setUp(self):
    stack = self.__stack = contextlib.ExitStack()
    super().setUp()
    try:
      slideids = "M206", "M148"

      testroot = thisfolder/"test_for_jenkins"/"geom"
      for SlideID in slideids:
        logfolder = testroot/SlideID/"logfiles"
        logfolder.mkdir(exist_ok=True, parents=True)

        filename = logfolder/f"{SlideID}-geom.log"
        assert stack.enter_context(job_lock.JobLock(filename))
        with getlogger(root=testroot, samp=SampleDef(SlideID=SlideID, Project=0, Cohort=0), module="geom", reraiseexceptions=False, uselogfiles=True, printthreshold=logging.CRITICAL+1):
          raise ValueError("testing error regex matching")

        dbloadfolder = testroot/SlideID/"dbload"
        dbloadfolder.mkdir(exist_ok=True, parents=True)
        for thing in "tumor", "field":
          (dbloadfolder/f"{SlideID}_{thing}Geometry.csv").touch()

    except:
      stack.close()
      raise

  def tearDown(self):
    super().tearDown()
    self.__stack.close()

  def testGeom(self, SlideID="M206", units="safe", selectrectangles=None):
    root = thisfolder/"data"
    testroot = thisfolder/"test_for_jenkins"/"geom"
    args = [os.fspath(root), "--dbloadroot", os.fspath(testroot), "--logroot", os.fspath(testroot), "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits", "--ignore-dependencies"]
    if selectrectangles is not None:
      args.append("--selectrectangles")
      for rid in selectrectangles: args.append(str(rid))

    s = GeomSample(root=thisfolder/"data", dbloadroot=testroot, samp=SlideID)
    tumorfilename = s.csv("tumorGeometry")
    fieldfilename = s.csv("fieldGeometry")
    reffolder = thisfolder/"data"/"reference"/"geom"/SlideID/"dbload"
    tumorreference = reffolder/tumorfilename.name
    fieldreference = reffolder/fieldfilename.name

    #this should not run anything
    GeomCohort.runfromargumentparser(args=args + ["--rerun-error", "anything but testing error regex matching"])
    with open(tumorfilename) as f:
      assert not f.read().strip()

    GeomCohort.runfromargumentparser(args=args + ["--rerun-error", "testing error regex matching"])

    try:
      rows = s.readtable(fieldfilename, Boundary, checkorder=True, checknewlines=True)
      targetrows = s.readtable(fieldreference, Boundary, checkorder=True, checknewlines=True)
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5)
        self.assertGreater(row.poly.area, 0)

      rows = s.readtable(tumorfilename, Boundary, checkorder=True, checknewlines=True)
      targetrows = s.readtable(tumorreference, Boundary, checkorder=True, checknewlines=True)
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5)
        self.assertGreater(row.poly.area, 0)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testGeomFastUnits(self, SlideID="M206", **kwargs):
    self.testGeom(SlideID=SlideID, units="fast", **kwargs)

  def testWithHoles(self, **kwargs):
    self.testGeom(SlideID="M148", selectrectangles=[15, 16], **kwargs)

  def testWithHolesFastUnits(self, **kwargs):
    self.testWithHoles(units="fast_microns", **kwargs)
