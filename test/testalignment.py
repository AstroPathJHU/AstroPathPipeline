import logging, more_itertools, numpy as np, os, pathlib, re, unittest
from astropath.shared.samplemetadata import SampleDef
from astropath.slides.align.aligncohort import AlignCohort
from astropath.slides.align.alignsample import AlignSample, AlignSampleComponentTiff, AlignSampleFromXML, ImageStats
from astropath.slides.align.overlap import AlignmentResult
from astropath.slides.align.field import Field, FieldOverlap
from astropath.slides.align.stitch import AffineEntry
from astropath.utilities import units
from astropath.utilities.misc import re_subs
from .testbase import assertAlmostEqual, expectedFailureIf, temporarilyremove, temporarilyreplace, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestAlignment(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    for SlideID in "M21_1", "YZ71", "M206":
      olddbload = thisfolder/"data"/SlideID/"dbload"
      newdbload = thisfolder/"test_for_jenkins"/"alignment"/SlideID/"dbload"
      newdbload2 = thisfolder/"test_for_jenkins"/"alignment"/"component_tiff"/SlideID/"dbload"
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
      thisfolder/"test_for_jenkins"/"alignment"/"M21_1"/"dbload"/filename.name
      for filename in (thisfolder/"data"/"reference"/"alignment"/"M21_1"/"dbload").glob("M21_1_*")
    ] + [
      thisfolder/"test_for_jenkins"/"alignment"/"YZ71"/"dbload"/filename.name
      for filename in (thisfolder/"data"/"reference"/"alignment"/"YZ71"/"dbload").glob("YZ71_*")
    ] + [
      thisfolder/"test_for_jenkins"/"alignment"/"component_tiff"/"M206"/"dbload"/filename.name
      for filename in (thisfolder/"data"/"reference"/"alignment"/"component_tiff"/"M206"/"dbload").glob("M206_*")
    ] + [
      thisfolder/"test_for_jenkins"/"alignment"/"logfiles"/"align.log",
      thisfolder/"test_for_jenkins"/"alignment"/"component_tiff"/"logfiles"/"align.log",
      thisfolder/"test_for_jenkins"/"alignment"/"M21_1"/"logfiles"/"M21_1-align.log",
      thisfolder/"test_for_jenkins"/"alignment"/"YZ71"/"logfiles"/"YZ71-align.log",
      thisfolder/"test_for_jenkins"/"alignment"/"component_tiff"/"M206"/"logfiles"/"M206-align.log",
    ]

  def testAlignment(self, SlideID="M21_1", componenttiff=False, **kwargs):
    samp = SampleDef(SlideID=SlideID, Project=0, Cohort=0, root=thisfolder/"data")
    dbloadroot = thisfolder/"test_for_jenkins"/"alignment"/("" if not componenttiff else "component_tiff")
    alignsampletype = AlignSample if not componenttiff else AlignSampleComponentTiff
    alignsampleargs = (
      thisfolder/"data",
      thisfolder/"data"/"flatw",
      samp,
    ) if not componenttiff else (
      thisfolder/"data",
      samp,
    )
    a = alignsampletype(
      *alignsampleargs,
      uselogfiles=True,
      dbloadroot=dbloadroot,
      logroot=dbloadroot,
      layer=1,
      **kwargs,
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
    finally:
      self.removeoutput()

  def compareoutput(self, alignsample, SlideID="M21_1", componenttiff=False):
    a = alignsample
    for filename, cls, extrakwargs in (
      (f"{SlideID}_imstat.csv", ImageStats, {}),
      (f"{SlideID}_align.csv", AlignmentResult, {}),
      (f"{SlideID}_fields.csv", Field, {}),
      (f"{SlideID}_affine.csv", AffineEntry, {}),
      (f"{SlideID}_fieldoverlaps.csv", FieldOverlap, {"rectangles": a.rectangles, "nclip": a.nclip}),
    ):
      testfolder = thisfolder/"test_for_jenkins"/"alignment"/("" if not componenttiff else "component_tiff")
      reffolder = thisfolder/"data"/"reference"/"alignment"/("" if not componenttiff else "component_tiff")
      try:
        rows = a.readtable(testfolder/SlideID/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True, checknewlines=True)
        targetrows = a.readtable(reffolder/SlideID/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True, checknewlines=True)
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
        from astropath.utilities.version import astropathversion
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
    a = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")
    agpu = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, useGPU=True, forceGPU=True, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")

    readfilename = thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/f"{SlideID}_align.csv"
    a.readalignments(filename=readfilename)

    with agpu:
      agpu.getDAPI(writeimstat=False)
      agpu.align(write_result=False)

    for o, ogpu in zip(a.overlaps, agpu.overlaps):
      assertAlmostEqual(o.result, ogpu.result, rtol=1e-5, atol=1e-5)

  def testReadAlignment(self, SlideID="M21_1"):
    try:
      a = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")
      readfilename = thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/f"{SlideID}_align.csv"
      writefilename = a.csv("align")

      a.readalignments(filename=readfilename)
      a.writealignments(filename=writefilename)
      rows = a.readtable(writefilename, AlignmentResult, checkorder=True, checknewlines=True)
      targetrows = a.readtable(readfilename, AlignmentResult, checkorder=True, checknewlines=True)
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5)
    except:
      self.saveoutput()
      raise
    finally:
      self.removeoutput()

  def testReadAlignmentFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast", "microns"):
      self.testReadAlignment(SlideID=SlideID)

  def testStitchReadingWriting(self, SlideID="M21_1"):
    try:
      a = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")
      a.readalignments(filename=thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/f"{SlideID}_align.csv")
      stitchfilenames = [thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/_.name for _ in a.stitchfilenames]
      result = a.readstitchresult(filenames=stitchfilenames)

      def referencefilename(filename): return thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/filename.name

      a.writestitchresult(result, filenames=a.stitchfilenames)

      for filename, cls, extrakwargs in more_itertools.zip_equal(
        a.stitchfilenames,
        (AffineEntry, Field, FieldOverlap),
        ({}, {}, {"nclip": a.nclip, "rectangles": a.rectangles}),
      ):
        rows = a.readtable(filename, cls, extrakwargs=extrakwargs, checkorder=True, checknewlines=True)
        targetrows = a.readtable(referencefilename(filename), cls, extrakwargs=extrakwargs, checkorder=True, checknewlines=True)
        for row, target in more_itertools.zip_equal(rows, targetrows):
          assertAlmostEqual(row, target, rtol=1e-5, atol=4e-7)
    except:
      self.saveoutput()
      raise
    finally:
      self.removeoutput()

  def testStitchReadingWritingFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testStitchReadingWriting(SlideID=SlideID)

  def testStitchWritingReading(self, SlideID="M21_1"):
    try:
      a1 = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")
      a1.readalignments(filename=thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/f"{SlideID}_align.csv")
      a1.stitch()

      a2 = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")
      a2.readalignments(filename=thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/f"{SlideID}_align.csv")
      stitchfilenames = [thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/_.name for _ in a1.stitchfilenames]
      a2.readstitchresult(filenames=stitchfilenames)

      pscale = a1.pscale
      assert pscale == a2.pscale

      rtol = 1e-6
      atol = 1e-6
      for o1, o2 in zip(a1.overlaps, a2.overlaps):
        x1, y1 = units.pixels(units.nominal_values(o1.stitchresult), pscale=pscale)
        x2, y2 = units.pixels(units.nominal_values(o2.stitchresult), pscale=pscale)
        assertAlmostEqual(x1, x2, rtol=rtol, atol=atol)
        assertAlmostEqual(y1, y2, rtol=rtol, atol=atol)

      for T1, T2 in zip(np.ravel(units.nominal_values(a1.T)), np.ravel(units.nominal_values(a2.T))):
        assertAlmostEqual(T1, T2, rtol=rtol, atol=atol)
    except:
      self.saveoutput()
      raise
    finally:
      self.removeoutput()

  def testStitchWritingReadingFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testStitchWritingReading(SlideID=SlideID)

  def testStitchCvxpy(self, SlideID="M21_1"):
    a = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")
    a.readalignments(filename=thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/f"{SlideID}_align.csv")

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
    a = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, selectrectangles=(10, 11), dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")
    with a:
      a.getDAPI(writeimstat=False)
      o1, o2 = a.overlaps
      o1.align()
      o2.align()
      assertAlmostEqual(o1.result.dx, -o2.result.dx, rtol=1e-5)
      assertAlmostEqual(o1.result.dy, -o2.result.dy, rtol=1e-5)
      assertAlmostEqual(o1.result.covxx, o2.result.covxx, rtol=1e-5)
      assertAlmostEqual(o1.result.covyy, o2.result.covyy, rtol=1e-5)
      assertAlmostEqual(o1.result.covxy, o2.result.covxy, rtol=1e-5)

  @unittest.skipIf(int(os.environ.get("JENKINS_PARALLEL", 0)), "temporarilyremove messes with other tests run in parallel")
  def testPscale(self, SlideID="M21_1"):
    a1 = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")
    readfilename = thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/f"{SlideID}_align.csv"
    stitchfilenames = [thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/_.name for _ in a1.stitchfilenames]
    a1.readalignments(filename=readfilename)
    a1.readstitchresult(filenames=stitchfilenames)

    constantsfile = a1.dbload/f"{SlideID}_constants.csv"
    with open(constantsfile) as f:
      constantscontents = f.read()
    newconstantscontents = constantscontents.replace(str(a1.pscale), str(a1.pscale * (1+1e-6)))
    assert newconstantscontents != constantscontents

    with temporarilyremove(thisfolder/"data"/SlideID/"inform_data"/"Component_Tiffs"), temporarilyreplace(constantsfile, newconstantscontents), temporarilyremove(thisfolder/"data"/SlideID/"im3"/"xml"):
      a2 = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")
      assert a1.pscale != a2.pscale
      with a2:
        a2.getDAPI(writeimstat=False)
        a2.align(debug=True)
        a2.stitch()

    pscale1 = a1.pscale
    pscale2 = a2.pscale
    rtol = 1e-5
    atol = 1e-7

    for o1, o2 in zip(a1.overlaps, a2.overlaps):
      x1, y1 = units.pixels(units.nominal_values(o1.stitchresult), pscale=pscale1)
      x2, y2 = units.pixels(units.nominal_values(o2.stitchresult), pscale=pscale2)
      assertAlmostEqual(x1, x2, rtol=rtol, atol=atol)
      assertAlmostEqual(y1, y2, rtol=rtol, atol=atol)

    for T1, T2 in zip(np.ravel(units.nominal_values(a1.T)), np.ravel(units.nominal_values(a2.T))):
      assertAlmostEqual(T1, T2, rtol=rtol, atol=atol)

  @unittest.skipIf(int(os.environ.get("JENKINS_PARALLEL", 0)), "temporarilyremove messes with other tests run in parallel")
  def testPscaleFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testPscale(SlideID=SlideID)

  def testCohort(self, units="safe"):
    SlideID = "M21_1"
    args = [os.fspath(thisfolder/"data"), "--shardedim3root", os.fspath(thisfolder/"data"/"flatw"), "--debug", "--dbloadroot", os.fspath(thisfolder/"test_for_jenkins"/"alignment"), "--logroot", os.fspath(thisfolder/"test_for_jenkins"/"alignment"), "--sampleregex", SlideID, "--units", units, "--allow-local-edits", "--ignore-dependencies", "--rerun-finished"]
    AlignCohort.runfromargumentparser(args)

    a = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")
    self.compareoutput(a)

  def testCohortFastUnits(self):
    self.testCohort(units="fast_microns")

  def testNoLog(self, SlideID="M21_1"):
    samp = SampleDef(SlideID=SlideID, Project=0, Cohort=0, root=thisfolder/"data")
    with AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", samp, selectrectangles=range(10), uselogfiles=True, logthreshold=logging.CRITICAL, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment") as a:
      a.getDAPI()
      a.align()
      a.stitch()

    for log in (
      thisfolder/"test_for_jenkins"/"alignment"/"logfiles"/"align.log",
      thisfolder/"test_for_jenkins"/"alignment"/SlideID/"logfiles"/f"{SlideID}-align.log",
    ):
      with open(log) as f:
        contents = f.read().splitlines()
      if len(contents) != 2:
        raise AssertionError(f"Expected only two lines of log\n\n{contents}")

  def testFromXML(self, SlideID="M21_1", **kwargs):
    args = thisfolder/"data", thisfolder/"data"/"flatw", SlideID
    kwargs = {**kwargs, "selectrectangles": range(10), "xmlfolders": [thisfolder/"data"/"raw"], "logroot": thisfolder/"test_for_jenkins"/"alignment"}
    a1 = AlignSample(*args, dbloadroot=thisfolder/"test_for_jenkins"/"alignment", **kwargs)
    with a1:
      a1.getDAPI()
      a1.align()
      result1 = a1.stitch()
      nclip = a1.nclip
      position = a1.position

    a2 = AlignSampleFromXML(*args, nclip=units.pixels(nclip, pscale=a1.pscale), position=position, **kwargs)
    with a2:
      a2.getDAPI()
      a2.align()
      result2 = a2.stitch()

    units.np.testing.assert_allclose(units.nominal_values(result1.T), units.nominal_values(result2.T))
    units.np.testing.assert_allclose(units.nominal_values(result1.x()), units.nominal_values(result2.x()))

    if not os.environ.get("JENKINS_PARALLEL", 0): #temporarilyremove messes with other tests run in parallel
      with temporarilyremove(thisfolder/"data"/SlideID/"dbload"), temporarilyremove(thisfolder/"data"/SlideID/"inform_data"):
        a3 = AlignSampleFromXML(*args, nclip=units.pixels(nclip, pscale=a1.pscale), **kwargs)
        with a3:
          a3.getDAPI()
          a3.align()
          result3 = a3.stitch()

      units.np.testing.assert_allclose(units.nominal_values(result1.T), units.nominal_values(result3.T), atol=1e-8)

  def testReadingLayer(self, SlideID="M21_1"):
    args = thisfolder/"data", thisfolder/"data"/"flatw", SlideID
    kwargs = {"selectrectangles": [17], "dbloadroot": thisfolder/"test_for_jenkins"/"alignment", "logroot": thisfolder/"test_for_jenkins"/"alignment"}
    a1 = AlignSample(*args, **kwargs)
    a2 = AlignSample(*args, **kwargs, readlayerfile=False, layer=1)
    i1 = a1.rectangles[0].alignmentimage
    i2 = a2.rectangles[0].alignmentimage
    np.testing.assert_array_equal(i1, i2)

  def testPolaris(self):
    self.testAlignment(SlideID="YZ71", xmlfolders=[thisfolder/"data"/"raw"])

  def testPolarisFastUnits(self):
    with units.setup_context("fast_microns"):
      self.testAlignment(SlideID="YZ71", xmlfolders=[thisfolder/"data"/"raw"])

  def testPolarisFromXMLFastUnits(self):
    with units.setup_context("fast"):
      self.testFromXML(SlideID="YZ71", xmlfolders=[thisfolder/"data"/"raw"])

  def testIslands(self, SlideID="M21_1"):
    for island in (
      #(4, 5),
      (5, 6),
      #(1, 2, 3, 5, 6, 7),
    ):
      a = AlignSample(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, selectoverlaps=lambda o: not ((o.p1 in island) ^ (o.p2 in island)), dbloadroot=thisfolder/"test_for_jenkins"/"alignment", logroot=thisfolder/"test_for_jenkins"/"alignment")
      readfilename = thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload"/f"{SlideID}_align.csv"
      a.readalignments(filename=readfilename)
      a.stitch()

  def testComponentTiff(self, SlideID="M206"):
    self.testAlignment(SlideID=SlideID, componenttiff=True)
