import dataclasses, itertools, numbers, numpy as np, os, uncertainties.unumpy as unp, unittest
from ..alignment.alignmentset import AlignmentSet, ImageStats
from ..alignment.overlap import AlignmentResult
from ..alignment.stitch import AffineEntry, StitchCoordinate, StitchOverlapCovariance
from ..utilities.tableio import readtable
from ..utilities import units

thisfolder = os.path.dirname(__file__)

def assertAlmostEqual(a, b, **kwargs):
  if isinstance(a, units.Distance):
    return units.testing.assert_allclose(a, b, **kwargs)
  elif isinstance(a, numbers.Number):
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
    for filename, cls, extrakwargs in (
      ("M21_1_imstat.csv", ImageStats, {"pscale": a.pscale}),
      ("M21_1_align.csv", AlignmentResult, {"pscale": a.pscale}),
      ("M21_1_stitch.csv", StitchCoordinate, {"pscale": a.pscale}),
      ("M21_1_affine.csv", AffineEntry, {}),
      ("M21_1_stitch_overlap_covariance.csv", StitchOverlapCovariance, {"pscale": a.pscale}),
    ):
      rows = readtable(os.path.join(thisfolder, "data", "M21_1", "dbload", filename), cls, extrakwargs=extrakwargs)
      targetrows = readtable(os.path.join(thisfolder, "alignmentreference", filename), cls, extrakwargs=extrakwargs)
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
    rows = readtable(writefilename, AlignmentResult, extrakwargs={"pscale": a.pscale})
    targetrows = readtable(readfilename, AlignmentResult, extrakwargs={"pscale": a.pscale})
    for row, target in itertools.zip_longest(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)

  def testStitchReadingWriting(self):
    a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1")
    a.readalignments(filename=os.path.join(thisfolder, "alignmentreference", "M21_1_align.csv"))
    result = a.readstitchresult()

    def newfilename(filename): return os.path.join(thisfolder, "test_"+os.path.basename(filename))

    a.writestitchresult(result, filenames=[newfilename(f) for f in a.stitchfilenames])

    for filename, cls, extrakwargs in itertools.zip_longest(
      a.stitchfilenames,
      (StitchCoordinate, AffineEntry, StitchOverlapCovariance),
      ({"pscale": a.pscale}, {}, {"pscale": a.pscale}),
    ):
      rows = readtable(newfilename(filename), cls, extrakwargs=extrakwargs)
      targetrows = readtable(filename, cls, extrakwargs=extrakwargs)
      for row, target in itertools.zip_longest(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5, atol=4e-7)

  def testStitchCvxpy(self):
    a = AlignmentSet(os.path.join(thisfolder, "data"), os.path.join(thisfolder, "data", "flatw"), "M21_1")
    a.readalignments(filename=os.path.join(thisfolder, "alignmentreference", "M21_1_align.csv"))

    defaultresult = a.stitch(saveresult=False)
    cvxpyresult = a.stitch(saveresult=False, usecvxpy=True)

    centerresult = a.stitch(saveresult=False, fixpoint="center")
    centercvxpyresult = a.stitch(saveresult=False, fixpoint="center", usecvxpy=True)

    units.testing.assert_allclose(centercvxpyresult.x(), units.nominal_values(centerresult.x()), rtol=1e-3)
    units.testing.assert_allclose(centercvxpyresult.T,   units.nominal_values(centerresult.T  ), rtol=1e-3, atol=1e-3)
    x = units.nominal_values(np.concatenate((centerresult.x(), centerresult.T), axis=None))
    units.testing.assert_allclose(
      centercvxpyresult.problem.value,
      x @ centerresult.A @ x + centerresult.b @ x + centerresult.c,
      rtol=0.1,
    )

    #test that the point you fix only affects the global translation
    units.testing.assert_allclose(
      cvxpyresult.x() - cvxpyresult.x()[0],
      centercvxpyresult.x() - centercvxpyresult.x()[0],
      rtol=1e-4,
    )
    units.testing.assert_allclose(
      cvxpyresult.problem.value,
      centercvxpyresult.problem.value,
      rtol=1e-4,
    )

    #test that the point you fix only affects the global translation
    units.testing.assert_allclose(
      units.nominal_values(defaultresult.x() - defaultresult.x()[0]),
      units.nominal_values(centerresult.x() - centerresult.x()[0]),
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
