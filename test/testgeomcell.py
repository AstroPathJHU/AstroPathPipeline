import csv, itertools, logging, more_itertools, os, pathlib, re

from astropath.shared.sample import SampleWithSegmentationFolder
from astropath.slides.geomcell.geomcellcohort import GeomCellCohortDeepCell, GeomCellCohortInform, GeomCellCohortMesmer
from astropath.slides.geomcell.geomcellsample import CellGeomLoad, GeomCellSampleDeepCell, GeomCellSampleInform, GeomCellSampleMesmer
from astropath.utilities.version.git import thisrepo

from .testbase import assertAlmostEqual, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestGeomCell(TestBaseSaveOutput):
  testrequirecommit = thisrepo.getcommit("cf271f3a")

  @property
  def outputfilenames(self):
    return [
      *(
        thisfolder/"test_for_jenkins"/"geomcell"/SlideID/"geom"/algo/filename.name
        for SlideID in ("M206", "M21_1")
        for algo in {
          "M206": ("inform",),
          "M21_1": ("deepcell", "mesmer"),
        }[SlideID]
        for filename in (thisfolder/"data"/"reference"/"geomcell"/SlideID/"geom"/algo).iterdir()
      ),
      *(
        thisfolder/"test_for_jenkins"/"geomcell"/"logfiles"/f"geomcell{algo}.log"
        for algo in ("", "deepcell", "mesmer")
      ),
      *(
        thisfolder/"test_for_jenkins"/"geomcell"/SlideID/"logfiles"/f"{SlideID}-geomcell{algo}.log"
        for SlideID in ("M206", "M21_1")
        for algo in ("", "deepcell", "mesmer")
      ),
    ]

  def testGeomCell(self, SlideID="M206", units="safe", algorithm="inform"):
    samplecls, cohortcls = {
      "inform": (GeomCellSampleInform, GeomCellCohortInform),
      "deepcell": (GeomCellSampleDeepCell, GeomCellCohortDeepCell),
      "mesmer": (GeomCellSampleMesmer, GeomCellCohortMesmer),
    }[algorithm]

    root = thisfolder/"data"
    geomroot = thisfolder/"test_for_jenkins"/"geomcell"
    segmentationroot = thisfolder/"data"/"reference"/"segmentation"
    args = [os.fspath(root), "--geomroot", os.fspath(geomroot), "--logroot", os.fspath(geomroot), "--selectrectangles", "1", "--units", units, "--sampleregex", SlideID, "--debug", "--allow-local-edits", "--ignore-dependencies", "--njobs", "2"]

    samplekwargs = {
      "root": root,
      "samp": SlideID,
      "geomroot": geomroot,
      "logroot": geomroot,
      "selectrectangles": [1],
      "printthreshold": logging.CRITICAL+1,
      "reraiseexceptions": False,
      "uselogfiles": True,
    }
    if issubclass(samplecls, SampleWithSegmentationFolder):
      samplekwargs["segmentationroot"] = segmentationroot
      args += ["--segmentationroot", os.fspath(segmentationroot)]
    s = samplecls(**samplekwargs)
    with s.logger:
      raise ValueError

    geomloadcsv = s.rectangles[0].geomloadcsv(algorithm)
    geomloadcsv.parent.mkdir(exist_ok=True, parents=True)
    geomloadcsv.touch()
    with s.logger:
      s.cleanup()
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
      contents = contents.replace(match.group("version"), f"{match.group('version')}.dev0+g{usecommit.shorthash(8)}", 2)
    else:
      contents = contents.replace(match.group("commit"), usecommit.shorthash(8), 2)

    with open(filename, "w", newline="") as f:
      f.write(contents)

    geomloadcsv.touch()
    #should not run anything because the csv already exists
    cohortcls.runfromargumentparser(args=args)
    #this shouldn't run anything either because the last cleanup was with the current commit
    cohortcls.runfromargumentparser(args=args + ["--require-commit", self.testrequirecommit.shorthash(8)])
    with open(geomloadcsv) as f:
      assert not f.read().strip()

    contents = contents.replace("Finished cleaning up", "")
    with open(filename, "w", newline="") as f:
      f.write(contents)
    #now there's no log message indicating a successful cleanup, so the last cleanup
    #was the first run of the log, which is testrequirecommit.parents[0]
    cohortcls.runfromargumentparser(args=args + ["--require-commit", self.testrequirecommit.shorthash(8)])

    try:
      for filename, reffilename in more_itertools.zip_equal(
        sorted(s.geomsubfolder.iterdir()),
        sorted((thisfolder/"data"/"reference"/"geomcell"/SlideID/"geom"/algorithm).iterdir()),
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

  def testGeomCellFastUnitsPixels(self, **kwargs):
    self.testGeomCell(units="fast_pixels", **kwargs)

  def testGeomCellDeepCell(self, **kwargs):
    self.testGeomCell(algorithm="deepcell", SlideID="M21_1", **kwargs)

  def testGeomCellMesmerFastUnitsMicrons(self, **kwargs):
    self.testGeomCell(algorithm="mesmer", SlideID="M21_1", units="fast_microns", **kwargs)
