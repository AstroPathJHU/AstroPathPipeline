import more_itertools, numpy as np, os, pathlib, PIL.Image, sys, unittest
from astropath_calibration.baseclasses.csvclasses import Annotation, Batch, Constant, QPTiffCsv, Region, ROIGlobals, Vertex
from astropath_calibration.baseclasses.overlap import Overlap
from astropath_calibration.baseclasses.rectangle import Rectangle
from astropath_calibration.prepdb.prepdbcohort import PrepdbCohort
from astropath_calibration.prepdb.prepdbsample import PrepdbSample
from astropath_calibration.utilities.tableio import readtable
from .testbase import assertAlmostEqual

thisfolder = pathlib.Path(__file__).parent

class TestPrepDb(unittest.TestCase):
  def setUp(self):
    self.maxDiff = None

  def testPrepDb(self, SlideID="M21_1", units="safe"):
    logs = (
      thisfolder/"data"/"logfiles"/"prepdb.log",
      thisfolder/"data"/SlideID/"logfiles"/f"{SlideID}-prepdb.log",
    )
    for log in logs:
      try:
        log.unlink()
      except FileNotFoundError:
        pass

    args = [os.fspath(thisfolder/"data"), "--sampleregex", SlideID, "--debug", "--units", units, "--xmlfolder", os.fspath(thisfolder/"data"/"raw"/SlideID)]
    PrepdbCohort.runfromargumentparser(args)
    sample = PrepdbSample(thisfolder/"data", SlideID, uselogfiles=False, xmlfolders=[thisfolder/"data"/"raw"/SlideID])

    for filename, cls, extrakwargs in (
      (f"{SlideID}_annotations.csv", Annotation, {"pscale": sample.pscale, "apscale": sample.apscale}),
      (f"{SlideID}_batch.csv", Batch, {}),
      (f"{SlideID}_constants.csv", Constant, {"pscale": sample.pscale, "apscale": sample.apscale, "qpscale": sample.qpscale, "readingfromfile": True}),
      (f"{SlideID}_globals.csv", ROIGlobals, {"pscale": sample.pscale}),
      (f"{SlideID}_overlap.csv", Overlap, {"pscale": sample.pscale, "nclip": sample.nclip, "rectangles": sample.rectangles}),
      (f"{SlideID}_qptiff.csv", QPTiffCsv, {"pscale": sample.pscale}),
      (f"{SlideID}_rect.csv", Rectangle, {"pscale": sample.pscale}),
      (f"{SlideID}_vertices.csv", Vertex, {"apscale": sample.apscale}),
      (f"{SlideID}_regions.csv", Region, {"apscale": sample.apscale, "pscale": sample.pscale}),
    ):
      if filename == "M21_1_globals.csv": continue
      try:
        rows = readtable(thisfolder/"data"/SlideID/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        targetrows = readtable(thisfolder/"reference"/"prepdb"/SlideID/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        for i, (row, target) in enumerate(more_itertools.zip_equal(rows, targetrows)):
          assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
      except:
        raise ValueError("Error in "+filename)

    with PIL.Image.open(thisfolder/"data"/SlideID/"dbload"/f"{SlideID}_qptiff.jpg") as img, \
         PIL.Image.open(thisfolder/"reference"/"prepdb"/SlideID/(f"{SlideID}_qptiff_windows.jpg" if sys.platform == "win32" else f"{SlideID}_qptiff.jpg")) as targetimg:
      np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))

  def testPrepDbFastUnits(self, SlideID="M21_1"):
    self.testPrepDb(SlideID, units="fast")

  @unittest.expectedFailure
  def testPrepDbPolaris(self):
    from .data.YZ71.im3.Scan3.assembleqptiff import assembleqptiff
    assembleqptiff()
    self.testPrepDb(SlideID="YZ71")
  @unittest.expectedFailure
  def testPrepDbPolarisFastUnits(self):
    self.testPrepDbFastUnits(SlideID="YZ71")

  def testPrepDbM206FastUnits(self):
    from .data.M206.im3.Scan1.assembleqptiff import assembleqptiff
    assembleqptiff()
    self.testPrepDbFastUnits(SlideID="M206")
