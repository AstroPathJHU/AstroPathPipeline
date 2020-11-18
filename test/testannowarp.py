import more_itertools, pathlib

from astropath_calibration.qptiff_alignment.qptiffalignmentsample import QPTiffAlignmentResult, QPTiffAlignmentSample, QPTiffStitchResultEntry
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

  def testAnnoWarp(self, SlideID="M206"):
    s = QPTiffAlignmentSample(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"annowarp_test_for_jenkins")
    s.align()
    filename = thisfolder/"annowarp_test_for_jenkins"/SlideID/s.alignmentcsv.name
    referencefilename = thisfolder/"reference"/"annowarp"/SlideID/s.alignmentcsv.name
    s.writealignments(filename=filename)

    rows = readtable(filename, QPTiffAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize})
    targetrows = readtable(referencefilename, QPTiffAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)

    s.stitch_cvxpy()
    filename = thisfolder/"annowarp_test_for_jenkins"/SlideID/s.stitchcsv.name
    referencefilename = thisfolder/"reference"/"annowarp"/SlideID/s.stitchcsv.name
    s.writestitchresult(filename=filename)

    rows = readtable(filename, QPTiffStitchResultEntry, extrakwargs={"pscale": s.pscale})
    targetrows = readtable(referencefilename, QPTiffStitchResultEntry, extrakwargs={"pscale": s.pscale})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)

  def testReadingWritingAlignments(self, SlideID="M206"):
    s = QPTiffAlignmentSample(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"annowarp_test_for_jenkins")
    referencefilename = thisfolder/"reference"/"annowarp"/SlideID/s.alignmentcsv.name
    testfilename = thisfolder/"annowarp_test_for_jenkins"/SlideID/"testreadannowarpalignments.csv"
    testfilename.parent.mkdir(parents=True, exist_ok=True)
    s.readalignments(filename=referencefilename)
    s.writealignments(filename=testfilename)
    rows = readtable(testfilename, QPTiffAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize})
    targetrows = readtable(referencefilename, QPTiffAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)
    testfilename.unlink()
