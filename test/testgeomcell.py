import more_itertools, pathlib

from astropath_calibration.geomcell.geomcellsample import CellGeomLoad, GeomCellSample
from astropath_calibration.utilities import units
from astropath_calibration.utilities.tableio import readtable

from .testbase import assertAlmostEqual, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestGeomCell(TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    return [
      thisfolder/"geomcell_test_for_jenkins"/SlideID/"geom"/filename.name
      for SlideID in ("M206",)
      for filename in (thisfolder/"reference"/"geomcell"/SlideID).iterdir()
    ]

  def testGeom(self, SlideID="M206", **kwargs):
    s = GeomCellSample(root=thisfolder/"data", samp=SlideID, geomroot=thisfolder/"geomcell_test_for_jenkins", selectrectangles=[1], **kwargs)
    s.rungeomcell()

    try:
      for filename, reffilename in more_itertools.zip_equal(
        sorted(s.geomfolder.iterdir()),
        sorted((thisfolder/"reference"/"geomcell"/SlideID).iterdir()),
      ):
        self.assertEqual(filename.name, reffilename.name)
  
        rows = readtable(filename, CellGeomLoad, extrakwargs={"pscale": s.pscale, "apscale": s.apscale})
        targetrows = readtable(reffilename, CellGeomLoad, extrakwargs={"pscale": s.pscale, "apscale": s.apscale})
        for row, target in more_itertools.zip_equal(rows, targetrows):
          assertAlmostEqual(row, target)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testGeomCellFastUnits(self, SlideID="M206", **kwargs):
    with units.setup_context("fast"):
      self.testGeom(SlideID=SlideID, **kwargs)
