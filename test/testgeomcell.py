import more_itertools, os, pathlib

from astropath.slides.geomcell.geomcellcohort import GeomCellCohort
from astropath.slides.geomcell.geomcellsample import CellGeomLoad, GeomCellSample

from .testbase import assertAlmostEqual, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestGeomCell(TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    return [
      thisfolder/"test_for_jenkins"/"geomcell"/SlideID/"geom"/filename.name
      for SlideID in ("M206",)
      for filename in (thisfolder/"data"/"reference"/"geomcell"/SlideID/"geom").iterdir()
    ]

  def testGeomCell(self, SlideID="M206", units="safe"):
    root = thisfolder/"data"
    geomroot = thisfolder/"test_for_jenkins"/"geomcell"
    args = [os.fspath(root), "--geomroot", os.fspath(geomroot), "--selectrectangles", "1", "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits", "--ignore-dependencies", "--njobs", "2"]
    s = GeomCellSample(root=root, samp=SlideID, geomroot=geomroot, selectrectangles=[1])

    filename = s.rectangles[0].geomloadcsv
    filename.parent.mkdir(exist_ok=True, parents=True)
    filename.touch()
    GeomCellCohort.runfromargumentparser(args=args)
    with open(s.rectangles[0].geomloadcsv) as f:
      assert not f.read().strip()
    s.samplelog.unlink()  #triggers cleanup
    GeomCellCohort.runfromargumentparser(args=args)

    try:
      for filename, reffilename in more_itertools.zip_equal(
        sorted(s.geomfolder.iterdir()),
        sorted((thisfolder/"data"/"reference"/"geomcell"/SlideID/"geom").iterdir()),
      ):
        self.assertEqual(filename.name, reffilename.name)
  
        rows = s.readtable(filename, CellGeomLoad, checkorder=True, checknewlines=True)
        targetrows = s.readtable(reffilename, CellGeomLoad, checkorder=True, checknewlines=True)
        for row, target in more_itertools.zip_equal(rows, targetrows):
          assertAlmostEqual(row, target)
          try:
            self.assertGreater(row.poly.area, -s.onepixel**2)
          except:
            s.printlogger.error(row)
            s.printlogger.error(row.poly.areas)
            raise
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testGeomCellFastUnitsPixels(self, SlideID="M206", **kwargs):
    self.testGeomCell(SlideID=SlideID, units="fast_pixels", **kwargs)

  def testGeomCellFastUnitsMicrons(self, SlideID="M206", **kwargs):
    self.testGeomCell(SlideID=SlideID, units="fast_microns", **kwargs)
