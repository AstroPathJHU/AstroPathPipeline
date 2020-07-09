import itertools, numpy as np, os, pathlib, PIL.Image, unittest
from ..baseclasses.sample import SampleDef
from ..prepdb.overlap import rectangleoverlaplist_fromcsvs
from ..prepdb.rectangle import Rectangle
from ..prepdb.prepdbsample import Annotation, Batch, Constant, Overlap, QPTiffCsv, Region, PrepdbSample, Vertex
from ..utilities import units
from ..utilities.tableio import readtable
from .testalignment import assertAlmostEqual

thisfolder = pathlib.Path(__file__).parent

class TestPrepDb(unittest.TestCase):
  def setUp(self):
    self.maxDiff = None

  def testPrepDb(self):
    logs = (
      thisfolder/"data"/"logfiles"/"prepdb.log",
      thisfolder/"data"/"M21_1"/"logfiles"/"M21_1-prepdb.log",
    )
    for log in logs:
      try:
        log.unlink()
      except FileNotFoundError:
        pass

    samp = SampleDef(SlideID="M21_1", SampleID=0, Project=0, Cohort=0, root=thisfolder/"data")
    sample = PrepdbSample(thisfolder/"data", samp, uselogfiles=True)
    with sample:
      sample.writemetadata()

    for filename, cls, extrakwargs in (
      ("M21_1_annotations.csv", Annotation, {}),
      ("M21_1_batch.csv", Batch, {}),
      ("M21_1_constants.csv", Constant, {"pscale": sample.pscale, "readingfromfile": True}),
      ("M21_1_overlap.csv", Overlap, {"pscale": sample.pscale, "layer": sample.layer, "nclip": sample.nclip, "rectangles": sample.rectangles}),
      ("M21_1_qptiff.csv", QPTiffCsv, {"pscale": sample.pscale}),
      ("M21_1_rect.csv", Rectangle, {"pscale": sample.pscale}),
      ("M21_1_regions.csv", Region, {"pscale": sample.pscale}),
      ("M21_1_vertices.csv", Vertex, {"pscale": sample.pscale}),
    ):
      try:
        rows = readtable(thisfolder/"data"/"M21_1"/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        targetrows = readtable(thisfolder/"reference"/"prepdb"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        for i, (row, target) in enumerate(itertools.zip_longest(rows, targetrows)):
          assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
      except:
        raise ValueError("Error in "+filename)

    with PIL.Image.open(thisfolder/"data"/"M21_1"/"dbload"/"M21_1_qptiff.jpg") as img, \
         PIL.Image.open(thisfolder/"reference"/"prepdb"/"M21_1_qptiff.jpg") as targetimg:
      np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))

      for log in logs:
        ref = thisfolder/"reference"/"prepdb"/log.name
        with open(ref) as fref, open(log) as fnew:
          refcontents = os.linesep.join([line.rsplit(";", 1)[0] for line in fref.read().splitlines() if "Biggest time difference" not in line])+os.linesep
          newcontents = os.linesep.join([line.rsplit(";", 1)[0] for line in fnew.read().splitlines() if "Biggest time difference" not in line])+os.linesep
          self.assertEqual(newcontents, refcontents)

  def testPrepDbFastUnits(self):
    with units.setup_context("fast"):
      self.testPrepDb()

  def testRectangleOverlapList(self):
    l = rectangleoverlaplist_fromcsvs(thisfolder/"data"/"M21_1"/"dbload")
    islands = l.islands()
    self.assertEqual(len(islands), 2)
    l2 = rectangleoverlaplist_fromcsvs(thisfolder/"data"/"M21_1"/"dbload", selectrectangles=lambda x: x.n in islands[0])
    self.assertEqual(l2.islands(), [l.islands()[0]])

  def testRectangleOverlapListFastUnits(self):
    with units.setup_context("fast"):
      self.testRectangleOverlapList()
