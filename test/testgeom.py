import more_itertools, os, pathlib

from astropath.slides.geom.geomcohort import GeomCohort
from astropath.slides.geom.geomsample import Boundary, GeomSample

from .testbase import assertAlmostEqual, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestGeom(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    for SlideID in "M206", "M148":
      olddbload = thisfolder/"data"/SlideID/"dbload"
      newdbload = thisfolder/"test_for_jenkins"/"geom"/SlideID/"dbload"
      for csv in (
        "constants",
        "fields",
      ):
        yield olddbload/f"{SlideID}_{csv}.csv", newdbload

  @property
  def outputfilenames(self):
    return [
      thisfolder/"test_for_jenkins"/"geom"/SlideID/"dbload"/f"{SlideID}_{csv}.csv"
      for csv in ("tumorGeometry", "fieldGeometry")
      for SlideID in ("M206", "M148")
    ]

  def testGeom(self, SlideID="M206", units="safe", selectrectangles=None):
    root = thisfolder/"data"
    dbloadroot = thisfolder/"test_for_jenkins"/"geom"
    args = [os.fspath(root), "--dbloadroot", os.fspath(dbloadroot), "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits", "--ignore-dependencies", "--rerun-finished"]
    if selectrectangles is not None:
      args.append("--selectrectangles")
      for rid in selectrectangles: args.append(str(rid))
    GeomCohort.runfromargumentparser(args=args)

    s = GeomSample(root=thisfolder/"data", dbloadroot=dbloadroot, samp=SlideID)
    tumorfilename = s.csv("tumorGeometry")
    fieldfilename = s.csv("fieldGeometry")
    reffolder = thisfolder/"data"/"reference"/"geom"/SlideID/"dbload"
    tumorreference = reffolder/tumorfilename.name
    fieldreference = reffolder/fieldfilename.name

    try:
      rows = s.readtable(fieldfilename, Boundary, checkorder=True, checknewlines=True)
      targetrows = s.readtable(fieldreference, Boundary, checkorder=True, checknewlines=True)
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5)
        self.assertGreater(row.poly.area, 0)

      rows = s.readtable(tumorfilename, Boundary, checkorder=True, checknewlines=True)
      targetrows = s.readtable(tumorreference, Boundary, checkorder=True, checknewlines=True)
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5)
        self.assertGreater(row.poly.area, 0)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testGeomFastUnits(self, SlideID="M206", **kwargs):
    self.testGeom(SlideID=SlideID, units="fast", **kwargs)

  def testWithHoles(self, **kwargs):
    self.testGeom(SlideID="M148", selectrectangles=[15, 16], **kwargs)

  def testWithHolesFastUnits(self, **kwargs):
    self.testWithHoles(units="fast_microns", **kwargs)
