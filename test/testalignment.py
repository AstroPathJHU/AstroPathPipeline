import dataclasses, itertools, numbers, numpy as np, os, uncertainties.unumpy as unp, unittest
from ..alignmentset import AlignmentSet, ImageStats, StitchCoordinate, StitchCovarianceEntry
from ..computeshift import crosscorrelation
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

  def testAlignment(self):
    a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1")
    a.getDAPI()
    a.align(debug=True, errorfactor=1/5)
    for filename, cls in (
      ("M21_1_imstat.csv", ImageStats),
      ("M21_1_align.csv", AlignmentResult),
      ("M21_1_stitch.csv", StitchCoordinate),
      ("M21_1_stitch_covariance.csv", StitchCovarianceEntry),
    ):
      rows = readtable(os.path.join(thisfolder, "data", "M21_1", "dbload", filename), cls)
      targetrows = readtable(os.path.join(thisfolder, "alignmentreference", filename), cls)
      for row, target in itertools.zip_longest(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5)

  def testReadAlignment(self):
    a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1")
    readfilename = os.path.join(thisfolder, "alignmentreference", "M21_1_align.csv")
    writefilename = os.path.join(thisfolder, "testreadalignments.csv")

    a.readalignments(filename=readfilename)
    a.writealignments(filename=writefilename)
    rows = readtable(writefilename, AlignmentResult)
    targetrows = readtable(readfilename, AlignmentResult)
    for row, target in itertools.zip_longest(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)

  def testStitchCvxpy(self):
    a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1")
    a.readalignments(filename=os.path.join(thisfolder, "alignmentreference", "M21_1_align.csv"))

    defaultresult = a.stitch()
    cvxpyresult = a.stitch(saveresult=False, usecvxpy=True)

    centerresult = a.stitch(saveresult=False, fixpoint="center")
    centercvxpyresult = a.stitch(saveresult=False, fixpoint="center", usecvxpy=True)

    np.testing.assert_allclose(centercvxpyresult.x(), unp.nominal_values(centerresult.x()), rtol=1e-3)
    np.testing.assert_allclose(centercvxpyresult.T,   unp.nominal_values(centerresult.T  ), rtol=1e-3, atol=1e-3)
    x = unp.nominal_values(np.concatenate((centerresult.x(), centerresult.T), axis=None))
    np.testing.assert_allclose(
      centercvxpyresult.problem.value,
      x @ centerresult.A @ x + centerresult.b @ x + centerresult.c,
      rtol=0.1,
    )

    #test that the point you fix only affects the global translation
    np.testing.assert_allclose(
      cvxpyresult.x() - cvxpyresult.x()[0],
      centercvxpyresult.x() - centercvxpyresult.x()[0],
      rtol=1e-4,
    )
    np.testing.assert_allclose(
      cvxpyresult.problem.value,
      centercvxpyresult.problem.value,
      rtol=1e-4,
    )

    #test that the point you fix only affects the global translation
    np.testing.assert_allclose(
      unp.nominal_values(defaultresult.x() - defaultresult.x()[0]),
      unp.nominal_values(centerresult.x() - centerresult.x()[0]),
      rtol=1e-4,
    )

  def testSymmetry(self):
    a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1", selectrectangles=(10, 11))
    a.getDAPI()
    o1, o2 = a.overlaps
    c1, c2 = crosscorrelation(o1.cutimages), crosscorrelation(o2.cutimages)
    np.testing.assert_allclose(np.roll(np.roll(np.fft.ifft2(c2).real[::-1,::-1], 1, axis=0), 1, axis=1), np.fft.ifft2(c1).real, rtol=1e-4)
