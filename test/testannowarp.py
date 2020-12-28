import more_itertools, pathlib

from astropath_calibration.annowarp.annowarpsample import AnnoWarpAlignmentResult, AnnoWarpSample, WarpedVertex
from astropath_calibration.annowarp.annowarpcohort import AnnoWarpCohort
from astropath_calibration.annowarp.stitch import AnnoWarpStitchResultEntry
from astropath_calibration.baseclasses.csvclasses import Region
from astropath_calibration.utilities import units
from astropath_calibration.utilities.tableio import readtable

from .testbase import assertAlmostEqual, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestAnnoWarp(TestBaseSaveOutput):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    from .data.M206.im3.Scan1.assembleqptiff import assembleqptiff
    assembleqptiff()

  @property
  def outputfilenames(self):
    return [
      thisfolder/"annowarp_test_for_jenkins"/SlideID/"warp_100.csv"
      for SlideID in ("M206",)
    ]

  def testAlignment(self, SlideID="M206"):
    s = AnnoWarpSample(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"annowarp_test_for_jenkins")
    s.align()
    alignmentfilename = thisfolder/"annowarp_test_for_jenkins"/SlideID/s.alignmentcsv.name
    referencealignmentfilename = thisfolder/"reference"/"annowarp"/SlideID/s.alignmentcsv.name
    s.writealignments(filename=alignmentfilename)

    s.stitch()
    stitchfilename = thisfolder/"annowarp_test_for_jenkins"/SlideID/s.stitchcsv.name
    referencestitchfilename = thisfolder/"reference"/"annowarp"/SlideID/s.stitchcsv.name
    s.writestitchresult(filename=stitchfilename)

    verticesfilename = thisfolder/"annowarp_test_for_jenkins"/SlideID/s.newverticescsv.name
    referenceverticesfilename = thisfolder/"reference"/"annowarp"/SlideID/s.newverticescsv.name
    s.writevertices(filename=verticesfilename)

    regionsfilename = thisfolder/"annowarp_test_for_jenkins"/SlideID/s.newregionscsv.name
    referenceregionsfilename = thisfolder/"reference"/"annowarp"/SlideID/s.newregionscsv.name
    s.writeregions(filename=regionsfilename)

    rows = readtable(alignmentfilename, AnnoWarpAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    targetrows = readtable(referencealignmentfilename, AnnoWarpAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)

    rows = readtable(stitchfilename, AnnoWarpStitchResultEntry, extrakwargs={"pscale": s.pscale})
    targetrows = readtable(referencestitchfilename, AnnoWarpStitchResultEntry, extrakwargs={"pscale": s.pscale})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)

    rows = readtable(verticesfilename, WarpedVertex, extrakwargs={"qpscale": s.imscale, "pscale": s.pscale, "bigtileoffset": s.bigtileoffset, "bigtilesize": s.bigtilesize})
    targetrows = readtable(referenceverticesfilename, WarpedVertex, extrakwargs={"qpscale": s.imscale, "pscale": s.pscale, "bigtileoffset": s.bigtileoffset, "bigtilesize": s.bigtilesize})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)

    rows = readtable(regionsfilename, Region, extrakwargs={"qpscale": s.imscale, "pscale": s.pscale})
    targetrows = readtable(referenceregionsfilename, Region, extrakwargs={"qpscale": s.imscale, "pscale": s.pscale})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)

  def testReadingWritingAlignments(self, SlideID="M206"):
    s = AnnoWarpSample(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"annowarp_test_for_jenkins")
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
    s = AnnoWarpSample(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"annowarp_test_for_jenkins")
    referencefilename = thisfolder/"reference"/"annowarp"/SlideID/s.alignmentcsv.name
    s.readalignments(filename=referencefilename)
    result1 = s.stitch()
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
    zoomroot = thisfolder/"annowarp_test_for_jenkins"
    args = [str(root), "--zoomroot", str(zoomroot), "--logroot", str(zoomroot), "--sampleregex", SlideID, "--debug", "--units", units]
    AnnoWarpCohort.runfromargumentparser(args)
