import itertools, pathlib, unittest
from ..alignment.rectangle import Rectangle
from ..prepdb.sample import Annotation, Constant, Overlap, QPTiffCsv, Region, Sample, Vertex
from ..utilities.tableio import readtable
from .testalignment import assertAlmostEqual

thisfolder = pathlib.Path(__file__).parent

class TestPrepDb(unittest.TestCase):
  def testPrepDb(self):
    sample = Sample(thisfolder/"data", "M21_1")
    sample.writemetadata()

    for filename, cls, extrakwargs in (
      ("M21_1_annotations.csv", Annotation, {}),
      ("M21_1_constants.csv", Constant, {}),
      ("M21_1_overlap.csv", Overlap, {"pscale": sample.pscale}),
      ("M21_1_qptiff.csv", QPTiffCsv, {"pscale": sample.pscale}),
      ("M21_1_rect.csv", Rectangle, {"pscale": sample.pscale}),
      ("M21_1_regions.csv", Region, {"pscale": sample.pscale}),
      ("M21_1_vertices.csv", Vertex, {"pscale": sample.pscale}),
    ):
      rows = readtable(thisfolder/"data"/"M21_1"/"dbload"/filename, cls, extrakwargs=extrakwargs)
      targetrows = readtable(thisfolder/"prepdbreference"/filename)
      for row, target in itertools.zip_longest(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)

