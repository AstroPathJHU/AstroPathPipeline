import more_itertools, numpy as np, pathlib, re

from astropath_calibration.annowarp.annowarpsample import AnnoWarpAlignmentResult, AnnoWarpSampleInformTissueMask, WarpedVertex
from astropath_calibration.annowarp.annowarpcohort import AnnoWarpCohort
from astropath_calibration.annowarp.stitch import AnnoWarpStitchResultEntry
from astropath_calibration.baseclasses.csvclasses import Region
from astropath_calibration.utilities import units
from astropath_calibration.utilities.tableio import readtable

from .testbase import assertAlmostEqual, TestBaseCopyInput, TestBaseSaveOutput

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
    return [
      thisfolder/"annowarp_test_for_jenkins"/SlideID/"warp_100.csv"
      for SlideID in ("M206",)
    ]

  def testAlignment(self, SlideID="M206"):
    s = AnnoWarpSampleInformTissueMask(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"reference"/"zoom", maskroot=thisfolder/"reference"/"stitchmask", dbloadroot=thisfolder/"annowarp_test_for_jenkins", logroot=thisfolder/"annowarp_test_for_jenkins", uselogfiles=True)

    alignmentfilename = s.alignmentcsv
    referencealignmentfilename = thisfolder/"reference"/"annowarp"/SlideID/s.alignmentcsv.name
    stitchfilename = s.stitchcsv
    referencestitchfilename = thisfolder/"reference"/"annowarp"/SlideID/s.stitchcsv.name
    verticesfilename = s.newverticescsv
    referenceverticesfilename = thisfolder/"reference"/"annowarp"/SlideID/s.newverticescsv.name
    regionsfilename = s.newregionscsv
    referenceregionsfilename = thisfolder/"reference"/"annowarp"/SlideID/s.newregionscsv.name

    with s:
      s.runannowarp()

    if not s.runstatus:
      raise ValueError(f"Annowarp on {s.SlideID} {s.runstatus}")

    rows = readtable(alignmentfilename, AnnoWarpAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    targetrows = readtable(referencealignmentfilename, AnnoWarpAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)

    rows = readtable(stitchfilename, AnnoWarpStitchResultEntry, extrakwargs={"pscale": s.pscale})
    targetrows = readtable(referencestitchfilename, AnnoWarpStitchResultEntry, extrakwargs={"pscale": s.pscale})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)

    rows = readtable(verticesfilename, WarpedVertex, extrakwargs={"apscale": s.apscale, "pscale": s.pscale, "bigtileoffset": s.bigtileoffset, "bigtilesize": s.bigtilesize})
    targetrows = readtable(referenceverticesfilename, WarpedVertex, extrakwargs={"apscale": s.apscale, "pscale": s.pscale, "bigtileoffset": s.bigtileoffset, "bigtilesize": s.bigtilesize})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)

    rows = readtable(regionsfilename, Region, extrakwargs={"apscale": s.apscale, "pscale": s.pscale})
    targetrows = readtable(referenceregionsfilename, Region, extrakwargs={"apscale": s.apscale, "pscale": s.pscale})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)
      self.assertGreater(row.poly.area, 0)

  def testReadingWritingAlignments(self, SlideID="M206"):
    s = AnnoWarpSampleInformTissueMask(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"reference"/"zoom", maskroot=thisfolder/"reference"/"stitchmask")
    referencefilename = thisfolder/"reference"/"annowarp"/SlideID/s.alignmentcsv.name
    testfilename = thisfolder/"annowarp_test_for_jenkins"/SlideID/"testreadannowarpalignments.csv"
    testfilename.parent.mkdir(parents=True, exist_ok=True)
    s.readalignments(filename=referencefilename)
    s.writealignments(filename=testfilename)
    rows = readtable(testfilename, AnnoWarpAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    targetrows = readtable(referencefilename, AnnoWarpAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)
    testfilename.unlink()

  def testStitchCvxpy(self, SlideID="M206"):
    s = AnnoWarpSampleInformTissueMask(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"reference"/"zoom", maskroot=thisfolder/"reference"/"stitchmask")
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
    args = [str(root), "--zoomroot", str(zoomroot), "--logroot", str(logroot), "--maskroot", str(maskroot), "--sampleregex", SlideID, "--debug", "--units", units, "--allow-local-edits"]
    AnnoWarpCohort.runfromargumentparser(args)

  def testConstraint(self, SlideID="M206"):
    s = AnnoWarpSampleInformTissueMask(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"reference"/"zoom", maskroot=thisfolder/"reference"/"stitchmask")
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
