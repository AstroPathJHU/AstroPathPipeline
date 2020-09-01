import itertools, numpy as np, os, pathlib, PIL.Image, unittest
from ..baseclasses.csvclasses import Annotation, Batch, Constant, Globals, QPTiffCsv, Region, Vertex
from ..baseclasses.sample import SampleDef
from ..baseclasses.overlap import Overlap, rectangleoverlaplist_fromcsvs
from ..baseclasses.rectangle import Rectangle
from ..prepdb.prepdbsample import PrepdbSample
from ..utilities import units
from ..utilities.tableio import readtable
from .testbase import assertAlmostEqual

thisfolder = pathlib.Path(__file__).parent

class TestPrepDb(unittest.TestCase):
  def setUp(self):
    self.maxDiff = None

  def testPrepDb(self, SlideID="M21_1"):
    logs = (
      thisfolder/"data"/"logfiles"/"prepdb.log",
      thisfolder/"data"/SlideID/"logfiles"/f"{SlideID}-prepdb.log",
    )
    for log in logs:
      try:
        log.unlink()
      except FileNotFoundError:
        pass

    samp = SampleDef(SlideID=SlideID, SampleID=0, Project=0, Cohort=0, root=thisfolder/"data")
    sample = PrepdbSample(thisfolder/"data", samp, uselogfiles=True)
    with sample:
      sample.writemetadata()

    for filename, cls, extrakwargs in (
      (f"{SlideID}_annotations.csv", Annotation, {}),
      (f"{SlideID}_batch.csv", Batch, {}),
      (f"{SlideID}_constants.csv", Constant, {"pscale": sample.pscale, "readingfromfile": True}),
      (f"{SlideID}_globals.csv", Globals, {"pscale": sample.pscale}),
      (f"{SlideID}_overlap.csv", Overlap, {"pscale": sample.pscale, "nclip": sample.nclip, "rectangles": sample.rectangles}),
      (f"{SlideID}_qptiff.csv", QPTiffCsv, {"pscale": sample.pscale}),
      (f"{SlideID}_rect.csv", Rectangle, {"pscale": sample.pscale}),
      (f"{SlideID}_regions.csv", Region, {"pscale": sample.pscale}),
      (f"{SlideID}_vertices.csv", Vertex, {"pscale": sample.pscale}),
    ):
      if filename == "M21_1_globals.csv": continue
      try:
        rows = readtable(thisfolder/"data"/SlideID/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        targetrows = readtable(thisfolder/"reference"/"prepdb"/SlideID/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        for i, (row, target) in enumerate(itertools.zip_longest(rows, targetrows)):
          assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
      except:
        raise ValueError("Error in "+filename)

    with PIL.Image.open(thisfolder/"data"/SlideID/"dbload"/f"{SlideID}_qptiff.jpg") as img, \
         PIL.Image.open(thisfolder/"reference"/"prepdb"/SlideID/f"{SlideID}_qptiff.jpg") as targetimg:
      np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))

      for log in logs:
        ref = thisfolder/"reference"/"prepdb"/SlideID/log.name
        with open(ref) as fref, open(log) as fnew:
          refcontents = os.linesep.join([line.rsplit(";", 1)[0] for line in fref.read().splitlines() if "Biggest time difference" not in line])+os.linesep
          newcontents = os.linesep.join([line.rsplit(";", 1)[0] for line in fnew.read().splitlines() if "Biggest time difference" not in line])+os.linesep
          self.assertEqual(newcontents, refcontents)

  def testPrepDbFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testPrepDb(SlideID)

  def testPrepDbPolaris(self):
    from .data.YZ71.im3.Scan3.assembleqptiff import assembleqptiff
    assembleqptiff()
    self.testPrepDb(SlideID="YZ71")
  def testPrepDbPolarisFastUnits(self):
    self.testPrepDbFastUnits(SlideID="YZ71")

  def testRectangleOverlapList(self):
    l = rectangleoverlaplist_fromcsvs(thisfolder/"data"/"M21_1"/"dbload")
    islands = l.islands()
    self.assertEqual(len(islands), 2)
    l2 = rectangleoverlaplist_fromcsvs(thisfolder/"data"/"M21_1"/"dbload", selectrectangles=lambda x: x.n in islands[0])
    self.assertEqual(l2.islands(), [l.islands()[0]])

  def testRectangleOverlapListFastUnits(self):
    with units.setup_context("fast"):
      self.testRectangleOverlapList()
