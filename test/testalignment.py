import itertools, logging, numpy as np, os, pathlib
from ..alignment.alignmentcohort import AlignmentCohort
from ..alignment.alignmentset import AlignmentSet, AlignmentSetFromXML, ImageStats
from ..alignment.overlap import AlignmentResult
from ..alignment.field import Field, FieldOverlap
from ..alignment.stitch import AffineEntry
from ..baseclasses.sample import SampleDef
from ..utilities.tableio import readtable
from ..utilities import units
from .testbase import assertAlmostEqual, expectedFailureIf, temporarilyremove, temporarilyreplace, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestAlignment(TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    return [
      thisfolder/"data"/"M21_1"/"dbload"/filename.name
      for filename in (thisfolder/"reference"/"alignment").glob("M21_1_*")
    ] + [
      thisfolder/"data"/"logfiles"/"align.log",
      thisfolder/"data"/"M21_1"/"logfiles"/"M21_1-align.log",
    ]

  def testAlignment(self):
    samp = SampleDef(SlideID="M21_1", SampleID=0, Project=0, Cohort=0)
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", samp, uselogfiles=True)
    with a:
      a.getDAPI()
      a.align(debug=True)
      a.stitch(checkwriting=True)

    try:
      self.compareoutput(a)
    finally:
      self.saveoutput()

  def compareoutput(self, alignmentset):
    a = alignmentset
    for filename, cls, extrakwargs in (
      ("M21_1_imstat.csv", ImageStats, {"pscale": a.pscale}),
      ("M21_1_align.csv", AlignmentResult, {"pscale": a.pscale}),
      ("M21_1_fields.csv", Field, {"pscale": a.pscale, "layer": a.layer}),
      ("M21_1_affine.csv", AffineEntry, {}),
      ("M21_1_fieldoverlaps.csv", FieldOverlap, {"pscale": a.pscale, "rectangles": a.rectangles, "nclip": a.nclip}),
    ):
      rows = readtable(thisfolder/"data"/"M21_1"/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
      targetrows = readtable(thisfolder/"reference"/"alignment"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
      for row, target in itertools.zip_longest(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)

    for log in (
      thisfolder/"data"/"logfiles"/"align.log",
      thisfolder/"data"/"M21_1"/"logfiles"/"M21_1-align.log",
    ):
      ref = thisfolder/"reference"/"alignment"/log.name
      with open(ref) as fref, open(log) as fnew:
        refcontents = os.linesep.join([line.rsplit(";", 1)[0] for line in fref.read().splitlines()])+os.linesep
        newcontents = os.linesep.join([line.rsplit(";", 1)[0] for line in fnew.read().splitlines()])+os.linesep
        self.assertEqual(refcontents, newcontents)

  def testAlignmentFastUnits(self):
    with units.setup_context("fast"):
      self.testAlignment()

  @expectedFailureIf(int(os.environ.get("JENKINS_NO_GPU", 0)))
  def testGPU(self):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    agpu = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1", useGPU=True, forceGPU=True)

    readfilename = thisfolder/"reference"/"alignment"/"M21_1_align.csv"
    a.readalignments(filename=readfilename)

    agpu.getDAPI(writeimstat=False)
    agpu.align(write_result=False)

    for o, ogpu in zip(a.overlaps, agpu.overlaps):
      assertAlmostEqual(o.result, ogpu.result, rtol=1e-5, atol=1e-5)

  def testReadAlignment(self):
    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    readfilename = thisfolder/"reference"/"alignment"/"M21_1_align.csv"
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
    a.readalignments(filename=thisfolder/"reference"/"alignment"/"M21_1_align.csv")
    stitchfilenames = [thisfolder/"reference"/"alignment"/_.name for _ in a.stitchfilenames]
    result = a.readstitchresult(filenames=stitchfilenames)

    def newfilename(filename): return thisfolder/("test_"+filename.name)
    def referencefilename(filename): return thisfolder/"reference"/"alignment"/filename.name

    a.writestitchresult(result, filenames=[newfilename(f) for f in a.stitchfilenames])

    for filename, cls, extrakwargs in itertools.zip_longest(
      a.stitchfilenames,
      (AffineEntry, Field, FieldOverlap),
      ({}, {"pscale": a.pscale, "layer": a.layer}, {"pscale": a.pscale, "nclip": a.nclip, "rectangles": a.rectangles}),
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
    a1.readalignments(filename=thisfolder/"reference"/"alignment"/"M21_1_align.csv")
    a1.stitch()

    a2 = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    a2.readalignments(filename=thisfolder/"reference"/"alignment"/"M21_1_align.csv")
    stitchfilenames = [thisfolder/"reference"/"alignment"/_.name for _ in a1.stitchfilenames]
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
    a.readalignments(filename=thisfolder/"reference"/"alignment"/"M21_1_align.csv")

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
    readfilename = thisfolder/"reference"/"alignment"/"M21_1_align.csv"
    stitchfilenames = [thisfolder/"reference"/"alignment"/_.name for _ in a1.stitchfilenames]
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

  def testCohort(self):
    cohort = AlignmentCohort(thisfolder/"data", thisfolder/"data"/"flatw", debug=True)
    cohort.run()

    a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    self.compareoutput(a)

  def testCohortFastUnits(self):
    with units.setup_context("fast"):
      self.testCohort()

  def testMissingFolders(self):
    with temporarilyremove(thisfolder/"data"/"M21_1"/"im3"), temporarilyremove(thisfolder/"data"/"M21_1"/"inform_data"), units.setup_context("fast"):
      a = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1", selectrectangles=range(10))
      a.getDAPI()
      a.align()
      a.stitch()

  def testNoLog(self):
    samp = SampleDef(SlideID="M21_1", SampleID=0, Project=0, Cohort=0)
    with AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", samp, selectrectangles=range(10), uselogfiles=True, logthreshold=logging.CRITICAL) as a:
      a.getDAPI()
      a.align()
      a.stitch()

    for log in (
      thisfolder/"data"/"logfiles"/"align.log",
      thisfolder/"data"/"M21_1"/"logfiles"/"M21_1-align.log",
    ):
      with open(log) as f:
        contents = f.read().splitlines()
      if len(contents) != 1:
        raise AssertionError(f"Expected only one line of log\n\n{contents}")

  def testFromXML(self):
    args = thisfolder/"data", thisfolder/"data"/"flatw", "M21_1"
    kwargs = {"selectrectangles": range(10)}
    a1 = AlignmentSet(*args, **kwargs)
    a1.getDAPI()
    a1.align()
    result1 = a1.stitch()

    with temporarilyremove(thisfolder/"data"/"M21_1"/"dbload"):
      a2 = AlignmentSetFromXML(*args, nclip=units.pixels(a1.nclip, pscale=a1.pscale), position=a1.position, **kwargs)
      a2.getDAPI()
      a2.align()
      result2 = a2.stitch()

      with temporarilyremove(thisfolder/"data"/"M21_1"/"inform_data"):
        a3 = AlignmentSetFromXML(*args, nclip=units.pixels(a1.nclip, pscale=a1.pscale), **kwargs)
        a3.getDAPI()
        a3.align()
        result3 = a3.stitch()

    units.np.testing.assert_allclose(units.nominal_values(result1.T), units.nominal_values(result2.T))
    units.np.testing.assert_allclose(units.nominal_values(result1.x()), units.nominal_values(result2.x()))
    units.np.testing.assert_allclose(units.nominal_values(result1.T), units.nominal_values(result3.T), atol=1e-8)

  def testReadingLayer(self):
    args = thisfolder/"data", thisfolder/"data"/"flatw", "M21_1"
    kwargs = {"selectrectangles": [17]}
    a1 = AlignmentSet(*args, **kwargs)
    a2 = AlignmentSet(*args, **kwargs, readlayerfile=False, layer=1)
    i1 = a1.rectangles[0].image
    i2 = a2.rectangles[0].image
    np.testing.assert_array_equal(i1, i2)
