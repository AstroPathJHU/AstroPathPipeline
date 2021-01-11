import more_itertools, pathlib

from astropath_calibration.geom.geomsample import Boundary, GeomSample
from astropath_calibration.utilities import units
from astropath_calibration.utilities.tableio import readtable

from .testbase import assertAlmostEqual, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestGeom(TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    return [
      thisfolder/"geom_test_for_jenkins"/SlideID/f"{SlideID}_{csv}.csv"
      for csv in ("tumorGeometry", "fieldGeometry")
      for SlideID in ("M206", "M148")
    ]

  def testGeom(self, SlideID="M206", **kwargs):
    s = GeomSample(root=thisfolder/"data", samp=SlideID, **kwargs)
    testfolder = thisfolder/"geom_test_for_jenkins"/SlideID
    testfolder.mkdir(parents=True, exist_ok=True)
    tumorfilename = testfolder/f"{SlideID}_tumorGeometry.csv"
    fieldfilename = testfolder/f"{SlideID}_fieldGeometry.csv"
    s.writeboundaries(tumorfilename=tumorfilename, fieldfilename=fieldfilename)

    reffolder = thisfolder/"reference"/"geom"/SlideID
    tumorreference = reffolder/tumorfilename.name
    fieldreference = reffolder/fieldfilename.name

    try:
      rows = readtable(fieldfilename, Boundary, extrakwargs={"pscale": s.pscale, "apscale": s.apscale})
      targetrows = readtable(fieldreference, Boundary, extrakwargs={"pscale": s.pscale, "apscale": s.apscale})
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5)
        self.assertGreater(row.poly.totalarea, 0)

      rows = readtable(tumorfilename, Boundary, extrakwargs={"pscale": s.pscale, "apscale": s.apscale})
      targetrows = readtable(tumorreference, Boundary, extrakwargs={"pscale": s.pscale, "apscale": s.apscale})
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5)
        self.assertGreater(row.poly.totalarea, 0)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testGeomFastUnits(self, SlideID="M206", **kwargs):
    with units.setup_context("fast"):
      self.testGeom(SlideID=SlideID, **kwargs)

  def testWithHoles(self):
    self.testGeom(SlideID="M148", selectrectangles=[15, 16])

  def testWithHolesFastUnits(self):
    with units.setup_context("fast"):
      self.testWithHoles()
