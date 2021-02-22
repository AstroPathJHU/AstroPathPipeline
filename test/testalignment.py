import contextlib2, logging, more_itertools, numpy as np, os, pathlib, re
from astropath_calibration.alignment.alignmentcohort import AlignmentCohort
from astropath_calibration.alignment.alignmentset import AlignmentSet, AlignmentSetComponentTiff, AlignmentSetFromXML, ImageStats
from astropath_calibration.alignment.overlap import AlignmentResult
from astropath_calibration.alignment.field import Field, FieldOverlap
from astropath_calibration.alignment.stitch import AffineEntry
from astropath_calibration.baseclasses.sample import SampleDef
from astropath_calibration.utilities import units
from astropath_calibration.utilities.misc import re_subs
from astropath_calibration.utilities.tableio import readtable
from .testbase import assertAlmostEqual, expectedFailureIf, temporarilyremove, temporarilyreplace, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestAlignment(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    for SlideID in "M21_1", "YZ71", "M206":
      olddbload = thisfolder/"data"/SlideID/"dbload"
      newdbload = thisfolder/"alignment_test_for_jenkins"/SlideID/"dbload"
      newdbload2 = thisfolder/"alignment_test_for_jenkins"/"component_tiff"/SlideID/"dbload"
      for csv in (
        "constants",
        "overlap",
        "rect",
      ):
        yield olddbload/f"{SlideID}_{csv}.csv", newdbload
        yield olddbload/f"{SlideID}_{csv}.csv", newdbload2

  @property
  def outputfilenames(self):
    return [
      thisfolder/"alignment_test_for_jenkins"/"M21_1"/"dbload"/filename.name
      for filename in (thisfolder/"reference"/"alignment"/"M21_1").glob("M21_1_*")
    ] + [
      thisfolder/"alignment_test_for_jenkins"/"YZ71"/"dbload"/filename.name
      for filename in (thisfolder/"reference"/"alignment"/"YZ71").glob("YZ71_*")
    ] + [
      thisfolder/"alignment_test_for_jenkins"/"component_tiff"/"M206"/"dbload"/filename.name
      for filename in (thisfolder/"reference"/"alignment"/"component_tiff"/"M206").glob("M206_*")
    ] + [
      thisfolder/"alignment_test_for_jenkins"/"logfiles"/"align.log",
      thisfolder/"alignment_test_for_jenkins"/"component_tiff"/"logfiles"/"align.log",
      thisfolder/"alignment_test_for_jenkins"/"M21_1"/"logfiles"/"M21_1-align.log",
      thisfolder/"alignment_test_for_jenkins"/"YZ71"/"logfiles"/"YZ71-align.log",
      thisfolder/"alignment_test_for_jenkins"/"component_tiff"/"M206"/"logfiles"/"M206-align.log",
    ]

  def testAlignment(self, SlideID="M21_1", componenttiff=False, **kwargs):
    samp = SampleDef(SlideID=SlideID, SampleID=0, Project=0, Cohort=0)
    dbloadroot = thisfolder/"alignment_test_for_jenkins"/("" if not componenttiff else "component_tiff")
    alignmentsettype = AlignmentSet if not componenttiff else AlignmentSetComponentTiff
    alignmentsetargs = (
      thisfolder/"data",
      thisfolder/"data"/"flatw",
      samp,
    ) if not componenttiff else (
      thisfolder/"data",
      samp,
    )
    a = alignmentsettype(
      *alignmentsetargs,
      uselogfiles=True,
      dbloadroot=dbloadroot,
      logroot=dbloadroot,
      **kwargs
    )
    with a:
      a.getDAPI()
      a.align()
      a.stitch(checkwriting=True)

    try:
      self.compareoutput(a, SlideID=SlideID, componenttiff=componenttiff)
    except:
      self.saveoutput()
      raise

  def compareoutput(self, alignmentset, SlideID="M21_1", componenttiff=False):
    a = alignmentset
    for filename, cls, extrakwargs in (
      (f"{SlideID}_imstat.csv", ImageStats, {"pscale": a.pscale}),
      (f"{SlideID}_align.csv", AlignmentResult, {"pscale": a.pscale}),
      (f"{SlideID}_fields.csv", Field, {"pscale": a.pscale}),
      (f"{SlideID}_affine.csv", AffineEntry, {}),
      (f"{SlideID}_fieldoverlaps.csv", FieldOverlap, {"pscale": a.pscale, "rectangles": a.rectangles, "nclip": a.nclip}),
    ):
      testfolder = thisfolder/"alignment_test_for_jenkins"/("" if not componenttiff else "component_tiff")
      reffolder = thisfolder/"reference"/"alignment"/("" if not componenttiff else "component_tiff")
      try:
        rows = readtable(testfolder/SlideID/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        targetrows = readtable(reffolder/SlideID/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        for row, target in more_itertools.zip_equal(rows, targetrows):
          if cls == AlignmentResult and row.exit != 0 and target.exit != 0: continue
          assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
      except:
        raise ValueError(f"Error in {filename}")

    for log in (
      testfolder/"logfiles"/"align.log",
      testfolder/SlideID/"logfiles"/f"{SlideID}-align.log",
    ):
      ref = reffolder/SlideID/log.name
      with open(ref) as fref, open(log) as fnew:
        subs = (";[^;]*$", ""), (r"(WARNING: (component tiff|xml files|constants\.csv)).*$", r"\1")
        from astropath_calibration.utilities.version import astropathversion
        refsubs = *subs, (r"(align )v[\w+.]+", rf"\1{astropathversion}")
        newsubs = *subs,
        refcontents = os.linesep.join([re_subs(line, *refsubs, flags=re.MULTILINE) for line in fref.read().splitlines()])+os.linesep
        newcontents = os.linesep.join([re_subs(line, *newsubs, flags=re.MULTILINE) for line in fnew.read().splitlines()])+os.linesep
        self.assertEqual(refcontents, newcontents)

  def testAlignmentFastUnits(self):
    with units.setup_context("fast"):
      self.testAlignment()

  @expectedFailureIf(int(os.environ.get("JENKINS_NO_GPU", 0)))
  def testGPU(self, SlideID="M21_1"):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
    agpu = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, useGPU=True, forceGPU=True, dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")

    readfilename = thisfolder/"reference"/"alignment"/SlideID/f"{SlideID}_align.csv"
    a.readalignments(filename=readfilename)

    agpu.getDAPI(writeimstat=False)
    agpu.align(write_result=False)

    for o, ogpu in zip(a.overlaps, agpu.overlaps):
      assertAlmostEqual(o.result, ogpu.result, rtol=1e-5, atol=1e-5)

  def testReadAlignment(self, SlideID="M21_1"):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
    readfilename = thisfolder/"reference"/"alignment"/SlideID/f"{SlideID}_align.csv"
    writefilename = thisfolder/"testreadalignments.csv"

    a.readalignments(filename=readfilename)
    a.writealignments(filename=writefilename)
    rows = readtable(writefilename, AlignmentResult, extrakwargs={"pscale": a.pscale})
    targetrows = readtable(readfilename, AlignmentResult, extrakwargs={"pscale": a.pscale})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)

  def testReadAlignmentFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testReadAlignment(SlideID=SlideID)

  def testStitchReadingWriting(self, SlideID="M21_1"):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
    a.readalignments(filename=thisfolder/"reference"/"alignment"/SlideID/f"{SlideID}_align.csv")
    stitchfilenames = [thisfolder/"reference"/"alignment"/SlideID/_.name for _ in a.stitchfilenames]
    result = a.readstitchresult(filenames=stitchfilenames)

    def newfilename(filename): return thisfolder/("test_"+filename.name)
    def referencefilename(filename): return thisfolder/"reference"/"alignment"/SlideID/filename.name

    a.writestitchresult(result, filenames=[newfilename(f) for f in a.stitchfilenames])

    for filename, cls, extrakwargs in more_itertools.zip_equal(
      a.stitchfilenames,
      (AffineEntry, Field, FieldOverlap),
      ({}, {"pscale": a.pscale}, {"pscale": a.pscale, "nclip": a.nclip, "rectangles": a.rectangles}),
    ):
      rows = readtable(newfilename(filename), cls, extrakwargs=extrakwargs)
      targetrows = readtable(referencefilename(filename), cls, extrakwargs=extrakwargs)
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5, atol=4e-7)

  def testStitchReadingWritingFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testStitchReadingWriting(SlideID=SlideID)

  def testStitchWritingReading(self, SlideID="M21_1"):
    a1 = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
    a1.readalignments(filename=thisfolder/"reference"/"alignment"/SlideID/f"{SlideID}_align.csv")
    a1.stitch()

    a2 = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
    a2.readalignments(filename=thisfolder/"reference"/"alignment"/SlideID/f"{SlideID}_align.csv")
    stitchfilenames = [thisfolder/"reference"/"alignment"/SlideID/_.name for _ in a1.stitchfilenames]
    a2.readstitchresult(filenames=stitchfilenames)

    pscale = a1.pscale
    assert pscale == a2.pscale

    rtol = 1e-6
    atol = 1e-6
    for o1, o2 in zip(a1.overlaps, a2.overlaps):
      x1, y1 = units.nominal_values(units.pixels(o1.stitchresult, pscale=pscale))
      x2, y2 = units.nominal_values(units.pixels(o2.stitchresult, pscale=pscale))
      assertAlmostEqual(x1, x2, rtol=rtol, atol=atol)
      assertAlmostEqual(y1, y2, rtol=rtol, atol=atol)

    for T1, T2 in zip(np.ravel(units.nominal_values(a1.T)), np.ravel(units.nominal_values(a2.T))):
      assertAlmostEqual(T1, T2, rtol=rtol, atol=atol)

  def testStitchWritingReadingFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testStitchWritingReading(SlideID=SlideID)

  def testStitchCvxpy(self, SlideID="M21_1"):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
    a.readalignments(filename=thisfolder/"reference"/"alignment"/SlideID/f"{SlideID}_align.csv")

    defaultresult = a.stitch(saveresult=False)
    cvxpyresult = a.stitch(saveresult=False, usecvxpy=True)

    centerresult = a.stitch(saveresult=False, fixpoint="center")
    centercvxpyresult = a.stitch(saveresult=False, fixpoint="center", usecvxpy=True)

    units.np.testing.assert_allclose(centercvxpyresult.x(), units.nominal_values(centerresult.x()), rtol=1e-3)
    units.np.testing.assert_allclose(centercvxpyresult.T,   units.nominal_values(centerresult.T  ), rtol=1e-3, atol=1e-3)
    x = units.nominal_values(np.concatenate((centerresult.x(), centerresult.T), axis=None))
    units.np.testing.assert_allclose(
      centercvxpyresult.problem.value,
      x @ centerresult.A @ x + centerresult.b @ x + centerresult.c,
      rtol=0.1,
    )

    #test that the point you fix only affects the global translation
    units.np.testing.assert_allclose(
      cvxpyresult.x() - cvxpyresult.x()[0],
      centercvxpyresult.x() - centercvxpyresult.x()[0],
      rtol=1e-4,
    )
    units.np.testing.assert_allclose(
      cvxpyresult.problem.value,
      centercvxpyresult.problem.value,
      rtol=1e-4,
    )

    #test that the point you fix only affects the global translation
    units.np.testing.assert_allclose(
      units.nominal_values(defaultresult.x() - defaultresult.x()[0]),
      units.nominal_values(centerresult.x() - centerresult.x()[0]),
      rtol=1e-4,
    )

  def testStitchCvxpyFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testStitchCvxpy(SlideID=SlideID)

  def testSymmetry(self, SlideID="M21_1"):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, selectrectangles=(10, 11), dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
    a.getDAPI(writeimstat=False)
    o1, o2 = a.overlaps
    o1.align()
    o2.align()
    assertAlmostEqual(o1.result.dx, -o2.result.dx, rtol=1e-5)
    assertAlmostEqual(o1.result.dy, -o2.result.dy, rtol=1e-5)
    assertAlmostEqual(o1.result.covxx, o2.result.covxx, rtol=1e-5)
    assertAlmostEqual(o1.result.covyy, o2.result.covyy, rtol=1e-5)
    assertAlmostEqual(o1.result.covxy, o2.result.covxy, rtol=1e-5)

  def testPscale(self, SlideID="M21_1"):
    a1 = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
    readfilename = thisfolder/"reference"/"alignment"/SlideID/f"{SlideID}_align.csv"
    stitchfilenames = [thisfolder/"reference"/"alignment"/SlideID/_.name for _ in a1.stitchfilenames]
    a1.readalignments(filename=readfilename)
    a1.readstitchresult(filenames=stitchfilenames)

    constantsfile = a1.dbload/f"{SlideID}_constants.csv"
    with open(constantsfile) as f:
      constantscontents = f.read()
    newconstantscontents = constantscontents.replace(str(a1.pscale), str(a1.pscale * (1+1e-6)))
    assert newconstantscontents != constantscontents

    with temporarilyremove(thisfolder/"data"/SlideID/"inform_data"/"Component_Tiffs"), temporarilyreplace(constantsfile, newconstantscontents), temporarilyremove(thisfolder/"data"/SlideID/"im3"/"xml"):
      a2 = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
      assert a1.pscale != a2.pscale
      a2.getDAPI(writeimstat=False)
      a2.align(debug=True)
      a2.stitch()

    pscale1 = a1.pscale
    pscale2 = a2.pscale
    rtol = 1e-5
    atol = 1e-7

    for o1, o2 in zip(a1.overlaps, a2.overlaps):
      x1, y1 = units.nominal_values(units.pixels(o1.stitchresult, pscale=pscale1))
      x2, y2 = units.nominal_values(units.pixels(o2.stitchresult, pscale=pscale2))
      assertAlmostEqual(x1, x2, rtol=rtol, atol=atol)
      assertAlmostEqual(y1, y2, rtol=rtol, atol=atol)

    for T1, T2 in zip(np.ravel(units.nominal_values(a1.T)), np.ravel(units.nominal_values(a2.T))):
      assertAlmostEqual(T1, T2, rtol=rtol, atol=atol)

  def testPscaleFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testPscale(SlideID=SlideID)

  def testCohort(self, units="safe"):
    SlideID = "M21_1"
    args = [str(thisfolder/"data"), str(thisfolder/"data"/"flatw"), "--debug", "--dbloadroot", str(thisfolder/"alignment_test_for_jenkins"), "--logroot", str(thisfolder/"alignment_test_for_jenkins"), "--sampleregex", SlideID, "--units", units]
    AlignmentCohort.runfromargumentparser(args)

    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
    self.compareoutput(a)

  def testCohortFastUnits(self):
    with units.setup_context("fast"):
      self.testCohort(units="fast")

  def testMissingFolders(self, SlideID="M21_1"):
    with temporarilyremove(thisfolder/"data"/SlideID/"im3"), temporarilyremove(thisfolder/"data"/SlideID/"inform_data"), units.setup_context("fast"):
      a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, selectrectangles=range(10), dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
      a.getDAPI()
      a.align()
      a.stitch()

  def testNoLog(self, SlideID="M21_1"):
    samp = SampleDef(SlideID=SlideID, SampleID=0, Project=0, Cohort=0)
    with AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", samp, selectrectangles=range(10), uselogfiles=True, logthreshold=logging.CRITICAL, dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins") as a:
      a.getDAPI()
      a.align()
      a.stitch()

    for log in (
      thisfolder/"alignment_test_for_jenkins"/"logfiles"/"align.log",
      thisfolder/"alignment_test_for_jenkins"/SlideID/"logfiles"/f"{SlideID}-align.log",
    ):
      with open(log) as f:
        contents = f.read().splitlines()
      if len(contents) != 1:
        raise AssertionError(f"Expected only one line of log\n\n{contents}")

  def testFromXML(self, SlideID="M21_1", **kwargs):
    args = thisfolder/"data", thisfolder/"data"/"flatw", SlideID
    kwargs = {**kwargs, "selectrectangles": range(10), "root3": thisfolder/"data"/"raw", "logroot": thisfolder/"alignment_test_for_jenkins"}
    a1 = AlignmentSet(*args, dbloadroot=thisfolder/"alignment_test_for_jenkins", **kwargs)
    a1.getDAPI()
    a1.align()
    result1 = a1.stitch()
    nclip = a1.nclip
    position = a1.position

    with contextlib2.nullcontext(): #temporarilyremove(thisfolder/"data"/SlideID/"dbload"):
      a2 = AlignmentSetFromXML(*args, nclip=units.pixels(nclip, pscale=a1.pscale), position=position, **kwargs)
      a2.getDAPI()
      a2.align()
      result2 = a2.stitch()

      """
      with temporarilyremove(thisfolder/"data"/SlideID/"inform_data"):
        a3 = AlignmentSetFromXML(*args, nclip=units.pixels(nclip, pscale=a1.pscale), **kwargs)
        a3.getDAPI()
        a3.align()
        result3 = a3.stitch()
      """

    units.np.testing.assert_allclose(units.nominal_values(result1.T), units.nominal_values(result2.T))
    units.np.testing.assert_allclose(units.nominal_values(result1.x()), units.nominal_values(result2.x()))
    #units.np.testing.assert_allclose(units.nominal_values(result1.T), units.nominal_values(result3.T), atol=1e-8)

  def testReadingLayer(self, SlideID="M21_1"):
    args = thisfolder/"data", thisfolder/"data"/"flatw", SlideID
    kwargs = {"selectrectangles": [17], "dbloadroot": thisfolder/"alignment_test_for_jenkins", "logroot": thisfolder/"alignment_test_for_jenkins"}
    a1 = AlignmentSet(*args, **kwargs)
    a2 = AlignmentSet(*args, **kwargs, readlayerfile=False, layer=1)
    i1 = a1.rectangles[0].image
    i2 = a2.rectangles[0].image
    np.testing.assert_array_equal(i1, i2)

  def testPolaris(self):
    self.testAlignment(SlideID="YZ71", root3=thisfolder/"data"/"raw")

  def testPolarisFastUnits(self):
    with units.setup_context("fast"):
      self.testAlignment(SlideID="YZ71", root3=thisfolder/"data"/"raw")

  def testPolarisFromXMLFastUnits(self):
    with units.setup_context("fast"):
      self.testFromXML(SlideID="YZ71", root3=thisfolder/"data"/"raw")

  def testIslands(self, SlideID="M21_1"):
    for island in (
      (4, 5),
      (5, 6),
      (1, 2, 3, 5, 6, 7),
    ):
      a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, selectoverlaps=lambda o: not ((o.p1 in island) ^ (o.p2 in island)), dbloadroot=thisfolder/"alignment_test_for_jenkins", logroot=thisfolder/"alignment_test_for_jenkins")
      readfilename = thisfolder/"reference"/"alignment"/SlideID/f"{SlideID}_align.csv"
      a.readalignments(filename=readfilename)
      a.stitch()

  def testComponentTiff(self, SlideID="M206"):
    self.testAlignment(SlideID=SlideID, componenttiff=True)
