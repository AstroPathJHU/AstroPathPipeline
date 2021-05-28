import contextlib, more_itertools, numpy as np, os, pathlib, re

from astropath.baseclasses.csvclasses import Region
from astropath.slides.annowarp.annowarpsample import AnnoWarpAlignmentResult, AnnoWarpSampleInformTissueMask, AnnoWarpSampleSelectMask, WarpedVertex
from astropath.slides.annowarp.detectbigshift import DetectBigShiftSample
from astropath.slides.annowarp.annowarpcohort import AnnoWarpCohortSelectMask
from astropath.slides.annowarp.stitch import AnnoWarpStitchResultEntry
from astropath.utilities import units

from .testbase import assertAlmostEqual, temporarilyremove, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestAnnoWarp(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    for SlideID in "M206",:
      olddbload = thisfolder/"data"/SlideID/"dbload"
      newdbload = thisfolder/"annowarp_test_for_jenkins"/SlideID/"dbload"
      for csv in (
        "constants",
        "vertices",
        "regions",
        "fields",
      ):
        yield olddbload/f"{SlideID}_{csv}.csv", newdbload

  @classmethod
  def removecopiedinput(cls): return False

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    from .data.M206.im3.Scan1.assembleqptiff import assembleqptiff
    assembleqptiff()
    from .testzoom import gunzipreference
    gunzipreference("M206")

  @property
  def outputfilenames(self):
    return sum(([
      thisfolder/"annowarp_test_for_jenkins"/SlideID/"annowarp.csv",
      thisfolder/"annowarp_test_for_jenkins"/SlideID/"annowarp-stitch.csv",
      thisfolder/"annowarp_test_for_jenkins"/SlideID/"regions.csv",
      thisfolder/"annowarp_test_for_jenkins"/SlideID/"vertices.csv",
    ] for SlideID in ("M206",)), [])

  def testAlignment(self, SlideID="M206"):
    s = AnnoWarpSampleInformTissueMask(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"reference"/"zoom", maskroot=thisfolder/"reference"/"stitchmask", dbloadroot=thisfolder/"annowarp_test_for_jenkins", logroot=thisfolder/"annowarp_test_for_jenkins", uselogfiles=True)

    alignmentfilename = s.alignmentcsv
    referencealignmentfilename = thisfolder/"reference"/"annowarp"/SlideID/s.alignmentcsv.name
    stitchfilename = s.stitchcsv
    referencestitchfilename = thisfolder/"reference"/"annowarp"/SlideID/s.stitchcsv.name
    verticesfilename = s.verticescsv
    referenceverticesfilename = thisfolder/"reference"/"annowarp"/SlideID/s.verticescsv.name
    regionsfilename = s.regionscsv
    referenceregionsfilename = thisfolder/"reference"/"annowarp"/SlideID/s.regionscsv.name

    AnnoWarpSampleSelectMask.runfromargumentparser([os.fspath(s.root), SlideID, "--zoomroot", os.fspath(s.zoomroot), "--maskroot", os.fspath(s.maskroot), "--dbloadroot", os.fspath(s.dbloadroot), "--logroot", os.fspath(s.logroot), "--inform-mask", "--allow-local-edits"])

    if not s.runstatus:
      raise ValueError(f"Annowarp on {s.SlideID} {s.runstatus}")

    rows = s.readtable(alignmentfilename, AnnoWarpAlignmentResult, extrakwargs={"tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    targetrows = s.readtable(referencealignmentfilename, AnnoWarpAlignmentResult, extrakwargs={"tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)

    rows = s.readtable(stitchfilename, AnnoWarpStitchResultEntry)
    targetrows = s.readtable(referencestitchfilename, AnnoWarpStitchResultEntry)
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)

    rows = s.readtable(verticesfilename, WarpedVertex, extrakwargs={"bigtileoffset": s.bigtileoffset, "bigtilesize": s.bigtilesize})
    targetrows = s.readtable(referenceverticesfilename, WarpedVertex, extrakwargs={"bigtileoffset": s.bigtileoffset, "bigtilesize": s.bigtilesize})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)

    rows = s.readtable(regionsfilename, Region)
    targetrows = s.readtable(referenceregionsfilename, Region)
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)
      self.assertGreater(row.poly.area, 0)

  def testReadingWritingAlignments(self, SlideID="M206"):
    s = AnnoWarpSampleInformTissueMask(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"reference"/"zoom", maskroot=thisfolder/"reference"/"stitchmask", dbloadroot=thisfolder/"annowarp_test_for_jenkins")
    referencefilename = thisfolder/"reference"/"annowarp"/SlideID/s.alignmentcsv.name
    testfilename = thisfolder/"annowarp_test_for_jenkins"/SlideID/"testreadannowarpalignments.csv"
    testfilename.parent.mkdir(parents=True, exist_ok=True)
    s.readalignments(filename=referencefilename)
    s.writealignments(filename=testfilename)
    rows = s.readtable(testfilename, AnnoWarpAlignmentResult, extrakwargs={"tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    targetrows = s.readtable(referencefilename, AnnoWarpAlignmentResult, extrakwargs={"tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)
    testfilename.unlink()

  def testStitchCvxpy(self, SlideID="M206"):
    s = AnnoWarpSampleInformTissueMask(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"reference"/"zoom", maskroot=thisfolder/"reference"/"stitchmask", dbloadroot=thisfolder/"annowarp_test_for_jenkins")
    referencefilename = thisfolder/"reference"/"annowarp"/SlideID/s.alignmentcsv.name
    s.readalignments(filename=referencefilename)
    result1 = s.stitch(residualpullcutoff=None)
    result2 = s.stitch_cvxpy()

    units.np.testing.assert_allclose(result2.coeffrelativetobigtile, units.nominal_values(result1.coeffrelativetobigtile))
    units.np.testing.assert_allclose(result2.bigtileindexcoeff, units.nominal_values(result1.bigtileindexcoeff))
    units.np.testing.assert_allclose(result2.constant, units.nominal_values(result1.constant))

    x = units.nominal_values(result1.flatresult)
    units.np.testing.assert_allclose(
      result2.problem.value,
      x @ result1.A @ x + result1.b @ x + result1.c,
      rtol=0.01,
    )

  def testCohort(self, SlideID="M206", units="fast"):
    root = thisfolder/"data"
    zoomroot = thisfolder/"reference"/"zoom"
    logroot = thisfolder/"annowarp_test_for_jenkins"
    maskroot = thisfolder/"reference"/"stitchmask"
    args = [os.fspath(root), "--zoomroot", os.fspath(zoomroot), "--logroot", os.fspath(logroot), "--maskroot", os.fspath(maskroot), "--sampleregex", SlideID, "--debug", "--units", units, "--allow-local-edits", "--dbloadroot", os.fspath(logroot), "--ignore-dependencies", "--rerun-finished", "--inform-mask"]
    with contextlib.ExitStack() as stack:
      for csv in "annotations", "regions", "vertices":
        stack.enter_context(temporarilyremove(root/SlideID/"dbload"/f"{SlideID}_{csv}.csv"))
      AnnoWarpCohortSelectMask.runfromargumentparser(args)

  def testConstraint(self, SlideID="M206"):
    s = AnnoWarpSampleInformTissueMask(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"reference"/"zoom", maskroot=thisfolder/"reference"/"stitchmask", dbloadroot=thisfolder/"annowarp_test_for_jenkins")
    referencefilename = thisfolder/"reference"/"annowarp"/SlideID/s.alignmentcsv.name
    s.readalignments(filename=referencefilename)
    result1 = s.stitch(residualpullcutoff=None)
    constraintmus = np.array([e.value for e in result1.stitchresultnominalentries])
    constraintsigmas = np.array([e.value for e in result1.stitchresultcovarianceentries if re.match(r"cov\((.*), \1\)", e.description)]) ** 0.5
    result2 = s.stitch(constraintmus=constraintmus, constraintsigmas=constraintsigmas, residualpullcutoff=None)
    result3 = s.stitch(constraintmus=constraintmus, constraintsigmas=constraintsigmas, residualpullcutoff=None, floatedparams="constants")

    units.np.testing.assert_allclose(units.nominal_values(result2.coeffrelativetobigtile), units.nominal_values(result1.coeffrelativetobigtile))
    units.np.testing.assert_allclose(units.nominal_values(result2.bigtileindexcoeff), units.nominal_values(result1.bigtileindexcoeff))
    units.np.testing.assert_allclose(units.nominal_values(result2.constant), units.nominal_values(result1.constant))
    units.np.testing.assert_allclose(units.nominal_values(result3.coeffrelativetobigtile), units.nominal_values(result1.coeffrelativetobigtile))
    units.np.testing.assert_allclose(units.nominal_values(result3.bigtileindexcoeff), units.nominal_values(result1.bigtileindexcoeff))
    units.np.testing.assert_allclose(units.nominal_values(result3.constant), units.nominal_values(result1.constant))
    x = units.nominal_values(result2.flatresult)
    units.np.testing.assert_allclose(
      x @ result3.A @ x + result3.b @ x + result3.c,
      x @ result2.A @ x + result2.b @ x + result2.c,
    )

    constraintmus *= 2
    result4 = s.stitch(constraintmus=constraintmus, constraintsigmas=constraintsigmas, residualpullcutoff=None)
    result5 = s.stitch_cvxpy(constraintmus=constraintmus, constraintsigmas=constraintsigmas)

    with np.testing.assert_raises(AssertionError):
      units.np.testing.assert_allclose(units.nominal_values(result2.coeffrelativetobigtile), units.nominal_values(result4.coeffrelativetobigtile))

    units.np.testing.assert_allclose(result5.coeffrelativetobigtile, units.nominal_values(result4.coeffrelativetobigtile))
    units.np.testing.assert_allclose(result5.bigtileindexcoeff, units.nominal_values(result4.bigtileindexcoeff))
    units.np.testing.assert_allclose(result5.constant, units.nominal_values(result4.constant))

    x = units.nominal_values(result4.flatresult)
    units.np.testing.assert_allclose(
      result5.problem.value,
      x @ result4.A @ x + result4.b @ x + result4.c,
      rtol=0.01,
    )

    result6 = s.stitch(constraintmus=constraintmus, constraintsigmas=constraintsigmas, residualpullcutoff=None, floatedparams="constants")
    units.np.testing.assert_allclose(units.nominal_values(result6.coeffrelativetobigtile)[:8], constraintmus[:4].reshape(2, 2))
    units.np.testing.assert_allclose(units.nominal_values(result6.bigtileindexcoeff), constraintmus[4:8].reshape(2, 2))

  def testDetectBigShift(self, SlideID="M21_1"):
    s = DetectBigShiftSample(root=thisfolder/"data", root2=thisfolder/"data"/"flatw", samp=SlideID, logroot=thisfolder/"annowarp_test_for_jenkins", uselogfiles=False, selectrectangles=[1])
    assertAlmostEqual(
      units.convertpscale(s.run(), s.pscale/10, s.pscale),
      np.array({
        "M21_1": [4.276086656088661, 10.14277048484133],
      }[SlideID])*s.onepixel,
      rtol=1e-6
    )

  def testDetectBigShiftFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast_microns"):
      self.testDetectBigShift(SlideID=SlideID)
