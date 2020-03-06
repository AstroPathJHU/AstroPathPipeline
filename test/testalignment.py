import dataclasses, numbers, numpy as np, os, unittest
from ..alignmentset import AlignmentSet, ImageStats, StitchCoordinate, StitchCovarianceEntry
from ..overlap import AlignmentResult
from ..tableio import readtable

thisfolder = os.path.dirname(__file__)

def assertAlmostEqual(a, b, **kwargs):
  if isinstance(a, numbers.Number):
    return np.testing.assert_allclose(a, b, **kwargs)
  elif dataclasses.is_dataclass(type(a)) and type(a) == type(b):
    try:
      for field in dataclasses.fields(type(a)):
        assertAlmostEqual(getattr(a, field.name), getattr(b, field.name), **kwargs)
    except AssertionError:
      np.testing.assert_equal(a, b)
  else:
    return np.testing.assert_equal(a, b)

class TestAlignment(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testAlign(self):
    a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1")
    a.getDAPI()
    a.align(debug=False, errorfactor=1/5)
    a.stitch()

  def testAlignmentResults(self):
    for filename, cls in (
      ("M21_1_imstat.csv", ImageStats),
      ("M21_1_align.csv", AlignmentResult),
      ("M21_1_stitch.csv", StitchCoordinate),
      ("M21_1_stitch_covariance.csv", StitchCovarianceEntry),
    ):
      rows = readtable(os.path.join(thisfolder, "data", "M21_1", "dbload", filename), cls)
      targetrows = readtable(os.path.join(thisfolder, "alignmentreference", filename), cls)
      for row, target in itertools.zip_longest(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5, atol=1e-8)
