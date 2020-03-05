import dataclasses, numbers, numpy as np, os, uncertainties.unumpy as unp, unittest
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
  @classmethod
  def setUpClass(cls):
    a = cls.a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1")
    a.getDAPI()
    a.align(debug=True, errorfactor=1/5)
    cls.stitchresult = a.stitch()

  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testAlignmentResults(self):
    for filename, cls in (
      ("M21_1_imstat.csv", ImageStats),
      ("M21_1_align.csv", AlignmentResult),
      ("M21_1_stitch.csv", StitchCoordinate),
      ("M21_1_stitch_covariance.csv", StitchCovarianceEntry),
    ):
      rows = readtable(os.path.join(thisfolder, "data", "M21_1", "dbload", filename), cls)
      targetrows = readtable(os.path.join(thisfolder, "alignmentreference", filename), cls)
      for row, target in zip(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5)

  def testStitchCvxpy(self):
    cvxpyresult = self.a.stitch(saveresult=False, usecvxpy=True)

    np.testing.assert_allclose(cvxpyresult.x(), unp.nominal_values(self.stitchresult.x()))
    np.testing.assert_allclose(cvxpyresult.T,   unp.nominal_values(self.stitchresult.T  ))

    x = np.concatenate((self.stitchresult.x(), self.stitchresult.T), axis=None)
    np.testing.assert_close(
      cvxpyresult.problem.value,
      x @ self.stitchresult.A @ x + self.stitchresult.b @ x + self.stitchresult.c
    )
