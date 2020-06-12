import contextlib, dataclasses, itertools, numbers, numpy as np, os, pathlib, shutil, tempfile, unittest
from ..alignment.alignmentset import AlignmentSet, ImageStats
from ..alignment.overlap import AlignmentResult
from ..alignment.field import Field, FieldOverlap
from ..alignment.stitch import AffineEntry
from ..utilities.tableio import readtable
from ..utilities import units

thisfolder = pathlib.Path(__file__).parent

def assertAlmostEqual(a, b, **kwargs):
  if isinstance(a, units.safe.Distance):
    return units.np.testing.assert_allclose(a, b, **kwargs)
  elif isinstance(a, numbers.Number):
    if isinstance(b, units.safe.Distance): b = float(b)
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

@contextlib.contextmanager
def temporarilyremove(filepath):
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    tmppath = d/filepath.name
    shutil.move(filepath, tmppath)
    try:
      yield
    finally:
      shutil.move(tmppath, filepath)

@contextlib.contextmanager
def temporarilyreplace(filepath, temporarycontents):
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    tmppath = d/filepath.name
    shutil.move(filepath, tmppath)
    with open(filepath, "w") as f:
      f.write(temporarycontents)
    try:
      yield
    finally:
      shutil.move(tmppath, filepath)

class TestAlignment(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.__aligned = None

  @property
  def alignedfilenames(self):
    return [
      thisfolder/"data"/"M21_1"/"dbload"/filename.name
      for filename in (thisfolder/"alignmentreference").glob("M21_1_*")
    ]

  def __savealigned(self):
    if self.__aligned is not None: return
    type(self).__aligned = contextlib.ExitStack()
    self.__aligned.__enter__()
    for filename in self.alignedfilenames:
      self.__aligned.enter_context(temporarilyremove(filename))

  def setUp(self):
    pass

  def tearDown(self):
    for filename in self.alignedfilenames:
      try:
        filename.unlink()
      except FileNotFoundError:
        pass

  @classmethod
  def tearDownClass(cls):
    if cls.__aligned is not None:
      cls.__aligned.__exit__(None, None, None)

  def testAlignment(self):
    for log in thisfolder/"data"/"logfiles"/"align.log", thisfolder/"data"/"M21_1"/"logfiles"/"align.log":
      if log.exists(): log.unlink()

    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1", uselogfiles=True)
    a.getDAPI()
    a.align(debug=True)
    a.stitch(checkwriting=True)

    try:
      for filename, cls, extrakwargs in (
        ("M21_1_imstat.csv", ImageStats, {"pscale": a.pscale}),
        ("M21_1_align.csv", AlignmentResult, {"pscale": a.pscale}),
        ("M21_1_fields.csv", Field, {"pscale": a.pscale}),
        ("M21_1_affine.csv", AffineEntry, {}),
        ("M21_1_fieldoverlaps.csv", FieldOverlap, {"pscale": a.pscale, "rectangles": a.rectangles, "layer": a.layer, "nclip": a.nclip}),
      ):
        rows = readtable(thisfolder/"data"/"M21_1"/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        targetrows = readtable(thisfolder/"alignmentreference"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        for row, target in itertools.zip_longest(rows, targetrows):
          assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
    finally:
      self.__savealigned()

    for ref, new in (
      (thisfolder/"alignmentreference"/"mainlog.log", thisfolder/"data"/"logfiles"/"align.log"),
      (thisfolder/"alignmentreference"/"samplelog.log", thisfolder/"data"/"M21_1"/"logfiles"/"align.log"),
    ):
      with open(ref) as fref, open(new) as fnew:
        refcontents = os.linesep.join([line.rsplit(",", 1)[0] for line in fref.read().splitlines()])+os.linesep
        newcontents = os.linesep.join([line.rsplit(",", 1)[0] for line in fnew.read().splitlines()])+os.linesep
        self.assertEqual(newcontents, refcontents)

  def testAlignmentFastUnits(self):
    with units.setup_context("fast"):
      self.testAlignment()

  @expectedFailureIf(int(os.environ.get("JENKINS_NO_GPU", 0)))
  def testGPU(self):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    agpu = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1", useGPU=True, forceGPU=True)

    readfilename = thisfolder/"alignmentreference"/"M21_1_align.csv"
    a.readalignments(filename=readfilename)

    agpu.getDAPI(writeimstat=False)
    agpu.align(write_result=False)

    for o, ogpu in zip(a.overlaps, agpu.overlaps):
      assertAlmostEqual(o.result, ogpu.result, rtol=1e-5, atol=1e-5)

  def testReadAlignment(self):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    readfilename = thisfolder/"alignmentreference"/"M21_1_align.csv"
    writefilename = thisfolder/"testreadalignments.csv"

    a.readalignments(filename=readfilename)
    a.writealignments(filename=writefilename)
    rows = readtable(writefilename, AlignmentResult, extrakwargs={"pscale": a.pscale})
    targetrows = readtable(readfilename, AlignmentResult, extrakwargs={"pscale": a.pscale})
    for row, target in itertools.zip_longest(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)

  def testReadAlignmentFastUnits(self):
    with units.setup_context("fast"):
      self.testReadAlignment()

  def testStitchReadingWriting(self):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    a.readalignments(filename=thisfolder/"alignmentreference"/"M21_1_align.csv")
    stitchfilenames = [thisfolder/"alignmentreference"/_.name for _ in a.stitchfilenames]
    result = a.readstitchresult(filenames=stitchfilenames)

    def newfilename(filename): return thisfolder/("test_"+filename.name)
    def referencefilename(filename): return thisfolder/"alignmentreference"/filename.name

    a.writestitchresult(result, filenames=[newfilename(f) for f in a.stitchfilenames])

    for filename, cls, extrakwargs in itertools.zip_longest(
      a.stitchfilenames,
      (AffineEntry, Field, FieldOverlap),
      ({}, {"pscale": a.pscale}, {"pscale": a.pscale, "layer": a.layer, "nclip": a.nclip, "rectangles": a.rectangles}),
    ):
      rows = readtable(newfilename(filename), cls, extrakwargs=extrakwargs)
      targetrows = readtable(referencefilename(filename), cls, extrakwargs=extrakwargs)
      for row, target in itertools.zip_longest(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5, atol=4e-7)

  def testStitchReadingWritingFastUnits(self):
    with units.setup_context("fast"):
      self.testStitchReadingWriting()

  def testStitchWritingReading(self):
    a1 = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    a1.readalignments(filename=thisfolder/"alignmentreference"/"M21_1_align.csv")
    a1.stitch()

    a2 = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    a2.readalignments(filename=thisfolder/"alignmentreference"/"M21_1_align.csv")
    stitchfilenames = [thisfolder/"alignmentreference"/_.name for _ in a1.stitchfilenames]
    a2.readstitchresult(filenames=stitchfilenames)

    pscale = a1.pscale
    assert pscale == a2.pscale

    rtol = 1e-6
    atol = 1e-6
    for o1, o2 in zip(a1.overlaps, a2.overlaps):
      x1, y1 = units.nominal_values(units.pixels(o1.stitchresult, pscale=pscale))
      x2, y2 = units.nominal_values(units.pixels(o2.stitchresult, pscale=pscale))
      assertAlmostEqual(x1, x2, rtol=rtol, atol=atol)
      assertAlmostEqual(y1, y2, rtol=rtol, atol=atol)

    for T1, T2 in zip(np.ravel(units.nominal_values(a1.T)), np.ravel(units.nominal_values(a2.T))):
      assertAlmostEqual(T1, T2, rtol=rtol, atol=atol)

  def testStitchWritingReadingFastUnits(self):
    with units.setup_context("fast"):
      self.testStitchWritingReading()

  def testStitchCvxpy(self):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    a.readalignments(filename=thisfolder/"alignmentreference"/"M21_1_align.csv")

    defaultresult = a.stitch(saveresult=False)
    cvxpyresult = a.stitch(saveresult=False, usecvxpy=True)

    centerresult = a.stitch(saveresult=False, fixpoint="center")
    centercvxpyresult = a.stitch(saveresult=False, fixpoint="center", usecvxpy=True)

    units.np.testing.assert_allclose(centercvxpyresult.x(), units.nominal_values(centerresult.x()), rtol=1e-3)
    units.np.testing.assert_allclose(centercvxpyresult.T,   units.nominal_values(centerresult.T  ), rtol=1e-3, atol=1e-3)
    x = units.nominal_values(np.concatenate((centerresult.x(), centerresult.T), axis=None))
    units.np.testing.assert_allclose(
      centercvxpyresult.problem.value,
      x @ centerresult.A @ x + centerresult.b @ x + centerresult.c,
      rtol=0.1,
    )

    #test that the point you fix only affects the global translation
    units.np.testing.assert_allclose(
      cvxpyresult.x() - cvxpyresult.x()[0],
      centercvxpyresult.x() - centercvxpyresult.x()[0],
      rtol=1e-4,
    )
    units.np.testing.assert_allclose(
      cvxpyresult.problem.value,
      centercvxpyresult.problem.value,
      rtol=1e-4,
    )

    #test that the point you fix only affects the global translation
    units.np.testing.assert_allclose(
      units.nominal_values(defaultresult.x() - defaultresult.x()[0]),
      units.nominal_values(centerresult.x() - centerresult.x()[0]),
      rtol=1e-4,
    )

  def testStitchCvxpyFastUnits(self):
    with units.setup_context("fast"):
      self.testStitchCvxpy()

  def testSymmetry(self):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1", selectrectangles=(10, 11))
    a.getDAPI(writeimstat=False)
    o1, o2 = a.overlaps
    o1.align()
    o2.align()
    assertAlmostEqual(o1.result.dx, -o2.result.dx, rtol=1e-5)
    assertAlmostEqual(o1.result.dy, -o2.result.dy, rtol=1e-5)
    assertAlmostEqual(o1.result.covxx, o2.result.covxx, rtol=1e-5)
    assertAlmostEqual(o1.result.covyy, o2.result.covyy, rtol=1e-5)
    assertAlmostEqual(o1.result.covxy, o2.result.covxy, rtol=1e-5)

  def testPscale(self):
    a1 = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    readfilename = thisfolder/"alignmentreference"/"M21_1_align.csv"
    stitchfilenames = [thisfolder/"alignmentreference"/_.name for _ in a1.stitchfilenames]
    a1.readalignments(filename=readfilename)
    a1.readstitchresult(filenames=stitchfilenames)

    constantsfile = a1.dbload/"M21_1_constants.csv"
    with open(constantsfile) as f:
      constantscontents = f.read()
    newconstantscontents = constantscontents.replace(str(a1.pscale), str(a1.pscale * (1+1e-6)))
    assert newconstantscontents != constantscontents

    with temporarilyremove(thisfolder/"data"/"M21_1"/"inform_data"/"Component_Tiffs"), temporarilyreplace(constantsfile, newconstantscontents):
      a2 = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
      assert a1.pscale != a2.pscale
      a2.getDAPI(writeimstat=False)
      a2.align(debug=True)
      a2.stitch()

    pscale1 = a1.pscale
    pscale2 = a2.pscale
    rtol = 1e-6
    atol = 1e-8

    for o1, o2 in zip(a1.overlaps, a2.overlaps):
      x1, y1 = units.nominal_values(units.pixels(o1.stitchresult, pscale=pscale1))
      x2, y2 = units.nominal_values(units.pixels(o2.stitchresult, pscale=pscale2))
      assertAlmostEqual(x1, x2, rtol=rtol, atol=atol)
      assertAlmostEqual(y1, y2, rtol=rtol, atol=atol)

    for T1, T2 in zip(np.ravel(units.nominal_values(a1.T)), np.ravel(units.nominal_values(a2.T))):
      assertAlmostEqual(T1, T2, rtol=rtol, atol=atol)

  def testPscaleFastUnits(self):
    with units.setup_context("fast"):
      self.testPscale()

