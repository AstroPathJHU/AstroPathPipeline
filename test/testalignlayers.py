import itertools, numpy as np, pathlib
from ..alignment.alignlayers import AlignLayers
from ..alignment.overlap import AlignmentResult, LayerAlignmentResult
from ..utilities import units
from ..utilities.tableio import readtable
from .testbase import assertAlmostEqual, expectedFailureIf, temporarilyremove, temporarilyreplace, TestBaseSaveOutput
thisfolder = pathlib.Path(__file__).parent

class TestAlignLayers(TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    return [
      thisfolder/"data"/"M21_1"/"dbload"/filename.name
      for filename in (thisfolder/"reference"/"alignlayers"/"M21_1").glob("M21_1_*")
    ] + [
      thisfolder/"data"/"YZ71"/"dbload"/filename.name
      for filename in (thisfolder/"reference"/"alignlayers"/"YZ71").glob("YZ71_*")
    ] + [
      thisfolder/"data"/"logfiles"/"alignlayers.log",
      thisfolder/"data"/"M21_1"/"logfiles"/"M21_1-alignlayers.log",
      thisfolder/"data"/"YZ71"/"logfiles"/"YZ71-alignlayers.log",
    ]

  def testAlignLayers(self, SlideID="M21_1"):
    a = AlignLayers(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, layers=range(1, 5), selectrectangles=(17,), use_mean_image=False)
    a.getDAPI()
    a.align()
    a.stitch(eliminatelayer=1)

    try:
      for filename, cls, extrakwargs in (
        (f"{SlideID}_alignlayers.csv", LayerAlignmentResult, {"pscale": a.pscale}),
      ):
        rows = readtable(thisfolder/"data"/SlideID/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        targetrows = readtable(thisfolder/"reference"/"alignlayers"/SlideID/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        for row, target in itertools.zip_longest(rows, targetrows):
          if cls == LayerAlignmentResult and row.exit != 0 and target.exit != 0: continue
          assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
    finally:
      self.saveoutput()

  def testEliminateLayerInvariance(self, SlideID="M21_1"):
    a = AlignLayers(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, layers=range(1, 5), selectrectangles=(17,), use_mean_image=False)
    a.align()
    #a.readalignments(filename=thisfolder/"reference"/"alignlayers"/SlideID/f"{SlideID}_alignlayers.csv")
    result1 = a.stitch(eliminatelayer=0)
    result2 = a.stitch(eliminatelayer=1)
    units.np.testing.assert_allclose(units.nominal_values(result1.x), units.nominal_values(result2.x))
    units.np.testing.assert_allclose(units.covariance_matrix(np.ravel(result1.x)), units.covariance_matrix(np.ravel(result2.x)))

  def testStitchCvxpy(self, SlideID="M21_1"):
    a = AlignLayers(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, layers=range(1, 5), selectrectangles=(17,), use_mean_image=False)
    a.align()
    #a.readalignments(filename=thisfolder/"reference"/"alignlayers"/SlideID/f"{SlideID}_alignlayers.csv")

    defaultresult = a.stitch(saveresult=False, eliminatelayer=2)
    cvxpyresult = a.stitch(saveresult=False, usecvxpy=True)

    units.np.testing.assert_allclose(cvxpyresult.x, units.nominal_values(defaultresult.x), atol=1e-6, rtol=1e-6)
    x = units.nominal_values(np.ravel(defaultresult.x))
    """
    units.np.testing.assert_allclose(
      cvxpyresult.problem.value,
      x @ defaultresult.A @ x + defaultresult.b @ x + defaultresult.c,
      rtol=0.1,
    )
    """

  def testReadAlignlayers(self, SlideID="M21_1"):
    a = AlignLayers(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, layers=range(1, 5), selectrectangles=(17,), use_mean_image=False)
    readfilename = thisfolder/"reference"/"alignlayers"/SlideID/f"{SlideID}_alignlayers.csv"
    writefilename = thisfolder/"testreadalignlayers.csv"

    a.readalignments(filename=readfilename)
    a.writealignments(filename=writefilename)
    rows = readtable(writefilename, LayerAlignmentResult, extrakwargs={"pscale": a.pscale})
    targetrows = readtable(readfilename, LayerAlignmentResult, extrakwargs={"pscale": a.pscale})
    for row, target in itertools.zip_longest(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)
