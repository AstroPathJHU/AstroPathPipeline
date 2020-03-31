import argparse, dataclasses, itertools, numbers, numpy as np, os, uncertainties.unumpy as unp, unittest
from ..alignmentset import AlignmentSet, ImageStats
from ..computeshift import crosscorrelation
from ..overlap import AlignmentResult
from ..stitch import AffineEntry, StitchCoordinate, StitchOverlapCovariance
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

def expectedFailureIf(condition):
  if condition:
    return unittest.expectedFailure
  else:
    return lambda function: function

class TestAlignment(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testAlignment(self):
    a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1")
    a.getDAPI()
    a.align(debug=True)
    a.stitch(checkwriting=True)
    for filename, cls in (
      ("M21_1_imstat.csv", ImageStats),
      ("M21_1_align.csv", AlignmentResult),
      ("M21_1_stitch.csv", StitchCoordinate),
      ("M21_1_affine.csv", AffineEntry),
      ("M21_1_stitch_overlap_covariance.csv", StitchOverlapCovariance),
    ):
      rows = readtable(os.path.join(thisfolder, "data", "M21_1", "dbload", filename), cls)
      targetrows = readtable(os.path.join(thisfolder, "alignmentreference", filename), cls)
      for row, target in itertools.zip_longest(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)

  @expectedFailureIf(int(os.environ.get("JENKINS_NO_GPU", 0)))
  def testGPU(self):
    a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1")
    agpu = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1", useGPU=True, forceGPU=True)

    readfilename = os.path.join(thisfolder, "alignmentreference", "M21_1_align.csv")
    a.readalignments(filename=readfilename)

    agpu.getDAPI()
    agpu.align(write_result=False)

    for o, ogpu in zip(a.overlaps, agpu.overlaps):
      assertAlmostEqual(o.result, ogpu.result, rtol=1e-5, atol=1e-5)

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

  def testStitchReadingWriting(self):
    a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1")
    a.readalignments(filename=os.path.join(thisfolder, "alignmentreference", "M21_1_align.csv"))
    result = a.readstitchresult()

    def newfilename(filename): return os.path.join(thisfolder, "test_"+os.path.basename(filename))

    a.writestitchresult(result, filenames=[newfilename(f) for f in a.stitchfilenames])

    for filename, cls in itertools.zip_longest(
      a.stitchfilenames,
      (StitchCoordinate, AffineEntry, StitchOverlapCovariance)
    ):
      rows = readtable(newfilename(filename), cls)
      targetrows = readtable(filename, cls)
      for row, target in itertools.zip_longest(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5, atol=4e-7)

  def testStitchCvxpy(self):
    a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1")
    a.readalignments(filename=os.path.join(thisfolder, "alignmentreference", "M21_1_align.csv"))

    defaultresult = a.stitch(saveresult=False)
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
    o1.align()
    o2.align()
    assertAlmostEqual(o1.result.dx, -o2.result.dx, rtol=1e-5)
    assertAlmostEqual(o1.result.dy, -o2.result.dy, rtol=1e-5)
    assertAlmostEqual(o1.result.covxx, o2.result.covxx, rtol=1e-5)
    assertAlmostEqual(o1.result.covyy, o2.result.covyy, rtol=1e-5)
    assertAlmostEqual(o1.result.covxy, o2.result.covxy, rtol=1e-5)
