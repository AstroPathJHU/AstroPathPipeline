import more_itertools, numpy as np, os, pathlib, PIL.Image, unittest
from astropath.baseclasses.csvclasses import Annotation, Batch, Constant, ExposureTime, QPTiffCsv, Region, ROIGlobals, Vertex
from astropath.baseclasses.overlap import Overlap
from astropath.baseclasses.rectangle import Rectangle
from astropath.slides.prepdb.prepdbcohort import PrepDbCohort
from astropath.slides.prepdb.prepdbsample import PrepDbSample
from .testbase import assertAlmostEqual

thisfolder = pathlib.Path(__file__).parent

class TestPrepDb(unittest.TestCase):
  def setUp(self):
    self.maxDiff = None

  def testPrepDb(self, SlideID="M21_1", units="safe", skipannotations=False):
    logs = (
      thisfolder/"data"/"logfiles"/"prepdb.log",
      thisfolder/"data"/SlideID/"logfiles"/f"{SlideID}-prepdb.log",
    )
    for log in logs:
      try:
        log.unlink()
      except FileNotFoundError:
        pass

    args = [os.fspath(thisfolder/"data"), "--sampleregex", SlideID, "--debug", "--units", units, "--xmlfolder", os.fspath(thisfolder/"data"/"raw"/SlideID), "--allow-local-edits"]
    if skipannotations:
      args.append("--skip-annotations")
    PrepDbCohort.runfromargumentparser(args)
    sample = PrepDbSample(thisfolder/"data", SlideID, uselogfiles=False, xmlfolders=[thisfolder/"data"/"raw"/SlideID])

    for filename, cls, extrakwargs in (
      (f"{SlideID}_annotations.csv", Annotation, {}),
      (f"{SlideID}_batch.csv", Batch, {}),
      (f"{SlideID}_constants.csv", Constant, {}),
      (f"{SlideID}_exposures.csv", ExposureTime, {}),
      (f"{SlideID}_globals.csv", ROIGlobals, {}),
      (f"{SlideID}_overlap.csv", Overlap, {"nclip": sample.nclip, "rectangles": sample.rectangles}),
      (f"{SlideID}_qptiff.csv", QPTiffCsv, {}),
      (f"{SlideID}_rect.csv", Rectangle, {}),
      (f"{SlideID}_vertices.csv", Vertex, {}),
      (f"{SlideID}_regions.csv", Region, {}),
    ):
      if filename == "M21_1_globals.csv": continue
      if skipannotations and cls in (Annotation, Vertex, Region):
        self.assertFalse((thisfolder/"data"/SlideID/"dbload"/filename).exists())
        continue
      try:
        rows = sample.readtable(thisfolder/"data"/SlideID/"dbload"/filename, cls, checkorder=True, extrakwargs=extrakwargs)
        targetrows = sample.readtable(thisfolder/"reference"/"prepdb"/SlideID/filename, cls, checkorder=True, extrakwargs=extrakwargs)
        for i, (row, target) in enumerate(more_itertools.zip_equal(rows, targetrows)):
          assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
      except:
        raise ValueError("Error in "+filename)

    with PIL.Image.open(thisfolder/"data"/SlideID/"dbload"/f"{SlideID}_qptiff.jpg") as img, \
         PIL.Image.open(thisfolder/"reference"/"prepdb"/SlideID/f"{SlideID}_qptiff.jpg") as targetimg:
      np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))

  def testPrepDbFastUnits(self, SlideID="M21_1"):
    self.testPrepDb(SlideID, units="fast")

  def testPrepDbPolaris(self, **kwargs):
    from .data.YZ71.im3.Scan3.assembleqptiff import assembleqptiff
    assembleqptiff()
    self.testPrepDb(SlideID="YZ71", skipannotations=True, **kwargs)
  def testPrepDbPolarisFastUnits(self):
    self.testPrepDbPolaris(units="fast")

  def testPrepDbM206FastUnits(self):
    from .data.M206.im3.Scan1.assembleqptiff import assembleqptiff
    assembleqptiff()
    self.testPrepDb(SlideID="M206", units="fast_microns")
