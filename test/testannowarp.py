import more_itertools, pathlib

from astropath_calibration.annowarp.annowarpsample import AnnoWarpAlignmentResult, AnnoWarpSample
from astropath_calibration.annowarp.stitch import AnnoWarpStitchResultEntry
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
    s = AnnoWarpSample(root=thisfolder/"data", samp=SlideID, zoomroot=thisfolder/"annowarp_test_for_jenkins")
    s.align()
    filename = thisfolder/"annowarp_test_for_jenkins"/SlideID/s.alignmentcsv.name
    referencefilename = thisfolder/"reference"/"annowarp"/SlideID/s.alignmentcsv.name
    s.writealignments(filename=filename)

    rows = readtable(filename, AnnoWarpAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    targetrows = readtable(referencefilename, AnnoWarpAlignmentResult, extrakwargs={"pscale": s.pscale, "tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset})
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)

    s.stitch_cvxpy()
    filename = thisfolder/"annowarp_test_for_jenkins"/SlideID/s.stitchcsv.name
    referencefilename = thisfolder/"reference"/"annowarp"/SlideID/s.stitchcsv.name
    s.writestitchresult(filename=filename)

    rows = readtable(filename, AnnoWarpStitchResultEntry, extrakwargs={"pscale": s.pscale})
    targetrows = readtable(referencefilename, AnnoWarpStitchResultEntry, extrakwargs={"pscale": s.pscale})
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
