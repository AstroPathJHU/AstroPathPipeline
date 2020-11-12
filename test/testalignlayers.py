import itertools, numpy as np, pathlib
from astropath_calibration.alignment.alignlayers import AlignLayers
from astropath_calibration.alignment.overlap import LayerAlignmentResult
from astropath_calibration.alignment.stitchlayers import LayerPosition, LayerPositionCovariance
from astropath_calibration.utilities import units
from astropath_calibration.utilities.tableio import readtable
from .testbase import assertAlmostEqual, TestBaseCopyInput, TestBaseSaveOutput
thisfolder = pathlib.Path(__file__).parent

class TestAlignLayers(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    for SlideID in "M21_1", "YZ71":
      olddbload = thisfolder/"data"/SlideID/"dbload"
      newdbload = thisfolder/"alignlayers_test_for_jenkins"/SlideID/"dbload"
      for csv in (
        "constants",
        "overlap",
        "rect",
      ):
        yield olddbload/f"{SlideID}_{csv}.csv", newdbload

  @property
  def outputfilenames(self):
    return [
      thisfolder/"alignlayers_test_for_jenkins"/"M21_1"/"dbload"/filename.name
      for filename in (thisfolder/"reference"/"alignlayers"/"M21_1").glob("M21_1_*")
    ] + [
      thisfolder/"alignlayers_test_for_jenkins"/"YZ71"/"dbload"/filename.name
      for filename in (thisfolder/"reference"/"alignlayers"/"YZ71").glob("YZ71_*")
    ] + [
      thisfolder/"alignlayers_test_for_jenkins"/"logfiles"/"alignlayers.log",
      thisfolder/"alignlayers_test_for_jenkins"/"M21_1"/"logfiles"/"M21_1-alignlayers.log",
      thisfolder/"alignlayers_test_for_jenkins"/"YZ71"/"logfiles"/"YZ71-alignlayers.log",
    ]

  def testAlignLayers(self, SlideID="M21_1"):
    a = AlignLayers(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, layers=range(1, 5), selectrectangles=(17, 23), use_mean_image=False, dbloadroot=thisfolder/"alignlayers_test_for_jenkins", logroot=thisfolder/"alignlayers_test_for_jenkins")
    a.getDAPI()
    a.align()
    a.stitch(eliminatelayer=1)

    try:
      for filename, cls, extrakwargs in (
        (f"{SlideID}_alignlayers.csv", LayerAlignmentResult, {"pscale": a.pscale}),
        (f"{SlideID}_layerpositions.csv", LayerPosition, {"pscale": a.pscale}),
        (f"{SlideID}_layerpositioncovariances.csv", LayerPositionCovariance, {"pscale": a.pscale}),
      ):
        rows = readtable(thisfolder/"alignlayers_test_for_jenkins"/SlideID/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        targetrows = readtable(thisfolder/"reference"/"alignlayers"/SlideID/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        for row, target in itertools.zip_longest(rows, targetrows):
          if cls == LayerAlignmentResult and row.exit != 0 and target.exit != 0: continue
          assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
    finally:
      self.saveoutput()

  def testAlignLayersFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testAlignLayers(SlideID=SlideID)

  def testEliminateLayerInvariance(self, SlideID="M21_1"):
    a = AlignLayers(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, layers=range(1, 5), selectrectangles=(17, 23), use_mean_image=False, dbloadroot=thisfolder/"alignlayers_test_for_jenkins", logroot=thisfolder/"alignlayers_test_for_jenkins")
    a.readalignments(filename=thisfolder/"reference"/"alignlayers"/SlideID/f"{SlideID}_alignlayers.csv")
    result1 = a.stitch(eliminatelayer=0)
    result2 = a.stitch(eliminatelayer=1)
    units.np.testing.assert_allclose(units.nominal_values(result1.x()), units.nominal_values(result2.x()), rtol=1e-7, atol=1e-7)
    units.np.testing.assert_allclose(units.covariance_matrix(np.ravel(result1.x())), units.covariance_matrix(np.ravel(result2.x())), rtol=1e-7, atol=1e-7)

  def testStitchCvxpy(self, SlideID="M21_1"):
    a = AlignLayers(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, layers=range(1, 5), selectrectangles=(17, 23), use_mean_image=False, dbloadroot=thisfolder/"alignlayers_test_for_jenkins", logroot=thisfolder/"alignlayers_test_for_jenkins")
    a.readalignments(filename=thisfolder/"reference"/"alignlayers"/SlideID/f"{SlideID}_alignlayers.csv")

    defaultresult = a.stitch(saveresult=False, eliminatelayer=0)
    cvxpyresult = a.stitch(saveresult=False, usecvxpy=True)

    units.np.testing.assert_allclose(cvxpyresult.x(), units.nominal_values(defaultresult.x()), atol=1e-6, rtol=1e-6)
    x = units.nominal_values(np.ravel(defaultresult.x()[1:]))
    units.np.testing.assert_allclose(
      cvxpyresult.problem.value,
      x @ defaultresult.A @ x + defaultresult.b @ x + defaultresult.c,
      atol=1e-7,
      rtol=1e-7,
    )

  def testReadAlignlayers(self, SlideID="M21_1"):
    a = AlignLayers(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, layers=range(1, 5), selectrectangles=(17, 23), use_mean_image=False, dbloadroot=thisfolder/"alignlayers_test_for_jenkins", logroot=thisfolder/"alignlayers_test_for_jenkins")
    readfilename = thisfolder/"reference"/"alignlayers"/SlideID/f"{SlideID}_alignlayers.csv"
    writefilename = thisfolder/"testreadalignlayers.csv"

    a.readalignments(filename=readfilename)
    a.writealignments(filename=writefilename)
    rows = readtable(writefilename, LayerAlignmentResult, extrakwargs={"pscale": a.pscale})
    targetrows = readtable(readfilename, LayerAlignmentResult, extrakwargs={"pscale": a.pscale})
    for row, target in itertools.zip_longest(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)

  def testBroadbandFilters(self, SlideID="M21_1"):
    a = AlignLayers(thisfolder/"data", thisfolder/"data"/"flatw", SlideID, selectrectangles=(17, 23), use_mean_image=False, dbloadroot=thisfolder/"alignlayers_test_for_jenkins", logroot=thisfolder/"alignlayers_test_for_jenkins")
    r = a.rectangles[0]
    np.testing.assert_array_equal(r.broadbandfilters, [1]*9+[2]*9+[3]*7+[4]*7+[5]*3)
