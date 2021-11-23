import csv, itertools, logging, more_itertools, os, pathlib, re

from astropath.slides.geomcell.geomcellcohort import GeomCellCohort
from astropath.slides.geomcell.geomcellsample import CellGeomLoad, GeomCellSample
from astropath.utilities.version.git import thisrepo

from .testbase import assertAlmostEqual, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestGeomCell(TestBaseSaveOutput):
  testrequirecommit = thisrepo.getcommit("cf271f3a")

  @property
  def outputfilenames(self):
    return [
      *(
        thisfolder/"test_for_jenkins"/"geomcell"/SlideID/"geom"/filename.name
        for SlideID in ("M206",)
        for filename in (thisfolder/"data"/"reference"/"geomcell"/SlideID/"geom").iterdir()
      ),
      thisfolder/"test_for_jenkins"/"geomcell"/"logfiles"/"geomcell.log",
      *(
        thisfolder/"test_for_jenkins"/"geomcell"/SlideID/"logfiles"/f"{SlideID}-geomcell.log"
        for SlideID in ("M206",)
      ),
    ]

  def testGeomCell(self, SlideID="M206", units="safe"):
    root = thisfolder/"data"
    geomroot = thisfolder/"test_for_jenkins"/"geomcell"
    args = [os.fspath(root), "--geomroot", os.fspath(geomroot), "--logroot", os.fspath(geomroot), "--selectrectangles", "1", "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits", "--ignore-dependencies", "--njobs", "2"]
    s = GeomCellSample(root=root, samp=SlideID, geomroot=geomroot, logroot=geomroot, selectrectangles=[1], printthreshold=logging.CRITICAL+1, reraiseexceptions=False, uselogfiles=True)
    with s.logger:
      raise ValueError

    filename = s.logger.samplelog
    with open(filename, newline="") as f:
      f, f2 = itertools.tee(f)
      startregex = re.compile(s.logstartregex())
      reader = csv.DictReader(f, fieldnames=("Project", "Cohort", "SlideID", "message", "time"), delimiter=";")
      for row in reader:
        match = startregex.match(row["message"])
        istag = not bool(match.group("commit"))
        if match: break
      else:
        assert False
      contents = "".join(f2)

    usecommit = self.testrequirecommit.parents[0]
    if istag:
      contents = contents.replace(match.group("version"), f"{match.group('version')}.dev0+g{usecommit.shorthash(8)}")
    else:
      contents = contents.replace(match.group("commit"), usecommit.shorthash(8))

    with open(filename, "w", newline="") as f:
      f.write(contents)

    filename = s.rectangles[0].geomloadcsv
    filename.parent.mkdir(exist_ok=True, parents=True)
    filename.touch()
    GeomCellCohort.runfromargumentparser(args=args)
    with open(s.rectangles[0].geomloadcsv) as f:
      assert not f.read().strip()
    GeomCellCohort.runfromargumentparser(args=args + ["--require-commit", self.testrequirecommit.shorthash(8)])

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
