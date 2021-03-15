import more_itertools, os, pathlib

from astropath_calibration.geomcell.geomcellcohort import GeomCellCohort
from astropath_calibration.geomcell.geomcellsample import CellGeomLoad, GeomCellSample
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

  def testGeomCell(self, SlideID="M206", units="safe"):
    root = thisfolder/"data"
    geomroot = thisfolder/"geomcell_test_for_jenkins"
    args = [os.fspath(root), "--geomroot", os.fspath(geomroot), "--selectrectangles", "1", "--units", units, "--sampleregex", SlideID, "--debug"]
    GeomCellCohort.runfromargumentparser(args=args)

    s = GeomCellSample(root=root, samp=SlideID, geomroot=geomroot, selectrectangles=[1])

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
          try:
            self.assertGreater(row.poly.area, -s.onepixel**2)
          except:
            print(row)
            print(row.poly.areas)
            raise
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testGeomCellFastUnits(self, SlideID="M206", **kwargs):
    self.testGeomCell(SlideID=SlideID, units="fast", **kwargs)
