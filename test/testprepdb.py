import itertools, numpy as np, pathlib, PIL.Image, unittest
from ..prepdb.overlap import rectangleoverlaplist_fromcsvs
from ..prepdb.rectangle import Rectangle
from ..prepdb.sample import Annotation, Batch, Constant, Overlap, QPTiffCsv, Region, Sample, Vertex
from ..utilities import units
from ..utilities.tableio import readtable
from .testalignment import assertAlmostEqual

thisfolder = pathlib.Path(__file__).parent

class TestPrepDb(unittest.TestCase):
  def testPrepDb(self):
    sample = Sample(thisfolder/"data", "M21_1")
    sample.writemetadata()

    for filename, cls, extrakwargs in (
      ("M21_1_annotations.csv", Annotation, {}),
      ("M21_1_batch.csv", Batch, {}),
      ("M21_1_constants.csv", Constant, {"pscale": sample.tiffpscale, "readingfromfile": True}),
      ("M21_1_overlap.csv", Overlap, {"pscale": sample.tiffpscale, "layer": sample.layer, "nclip": sample.nclip, "rectangles": sample.rectangles}),
      ("M21_1_qptiff.csv", QPTiffCsv, {"pscale": sample.tiffpscale}),
      ("M21_1_rect.csv", Rectangle, {"pscale": sample.tiffpscale}),
      ("M21_1_regions.csv", Region, {"pscale": sample.tiffpscale}),
      ("M21_1_vertices.csv", Vertex, {"pscale": sample.tiffpscale}),
    ):
      try:
        rows = readtable(thisfolder/"data"/"M21_1"/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        targetrows = readtable(thisfolder/"prepdbreference"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        for i, (row, target) in enumerate(itertools.zip_longest(rows, targetrows)):
          assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
      except:
        raise ValueError("Error in "+filename)

    with PIL.Image.open(thisfolder/"data"/"M21_1"/"dbload"/"M21_1_qptiff.jpg") as img, \
         PIL.Image.open(thisfolder/"prepdbreference"/"M21_1_qptiff.jpg") as targetimg:
      np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))

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
