import more_itertools, numpy as np, os, pathlib, PIL.Image
from astropath.shared.csvclasses import Annotation, Batch, Constant, ExposureTime, QPTiffCsv, Region, ROIGlobals, Vertex
from astropath.shared.overlap import Overlap
from astropath.shared.rectangle import Rectangle
from astropath.slides.prepdb.prepdbcohort import PrepDbCohort
from astropath.slides.prepdb.prepdbsample import PrepDbSample
from astropath.utilities.misc import checkwindowsnewlines
from .testbase import assertAlmostEqual, temporarilyreplace, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestPrepDb(TestBaseSaveOutput):
  def setUp(self):
    self.maxDiff = None

  @property
  def outputfilenames(self):
    SlideIDs = "M21_1", "YZ71", "M206"
    return [
      thisfolder/"test_for_jenkins"/"prepdb"/SlideID/"dbload"/filename.name
      for SlideID in SlideIDs
      for ext in ("csv", "jpg")
      for filename in (thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload").glob(f"*.{ext}")
    ] + [
      thisfolder/"test_for_jenkins"/"alignment"/SlideID/"logfiles"/f"{SlideID}-align.log"
      for SlideID in SlideIDs
    ] + [
      thisfolder/"test_for_jenkins"/"alignment"/"logfiles"/"align.log",
    ]


  def testPrepDb(self, SlideID="M21_1", units="safe", skipannotations=False):
    dbloadroot = thisfolder/"test_for_jenkins"/"prepdb"

    logs = (
      dbloadroot/"logfiles"/"prepdb.log",
      dbloadroot/SlideID/"logfiles"/f"{SlideID}-prepdb.log",
    )
    for log in logs:
      try:
        log.unlink()
      except FileNotFoundError:
        pass

    args = [os.fspath(thisfolder/"data"), "--sampleregex", SlideID, "--debug", "--units", units, "--xmlfolder", os.fspath(thisfolder/"data"/"raw"/SlideID), "--allow-local-edits", "--ignore-dependencies", "--rerun-finished", "--rename-annotation", "Good tisue", "Good tissue", "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(dbloadroot)]
    if skipannotations:
      args.append("--skip-annotations")

    try:
      PrepDbCohort.runfromargumentparser(args)
      sample = PrepDbSample(thisfolder/"data", SlideID, uselogfiles=False, xmlfolders=[thisfolder/"data"/"raw"/SlideID], dbloadroot=dbloadroot, logroot=dbloadroot)

      for filename, cls, extrakwargs in (
        (f"{SlideID}_annotations.csv", Annotation, {}),
        (f"{SlideID}_batch.csv", Batch, {}),
        (f"{SlideID}_constants.csv", Constant, {}),
        (f"{SlideID}_exposures.csv", ExposureTime, {}),
        (f"{SlideID}_globals.csv", ROIGlobals, {}),
        (f"{SlideID}_overlap.csv", Overlap, {"nclip": sample.nclip, "rectangles": sample.rectangles}),
        (f"{SlideID}_qptiff.csv", QPTiffCsv, {}),
        (f"{SlideID}_rect.csv", Rectangle, {}),
        (f"{SlideID}_vertices.csv", Vertex, {}),
        (f"{SlideID}_regions.csv", Region, {}),
      ):
        if filename == "M21_1_globals.csv": continue
        if skipannotations and cls in (Annotation, Vertex, Region):
          self.assertFalse((dbloadroot/SlideID/"dbload"/filename).exists())
          continue
        try:
          rows = sample.readtable(dbloadroot/SlideID/"dbload"/filename, cls, checkorder=True, checknewlines=True, extrakwargs=extrakwargs)
          targetrows = sample.readtable(thisfolder/"data"/"reference"/"prepdb"/SlideID/filename, cls, checkorder=True, checknewlines=True, extrakwargs=extrakwargs)
          for i, (row, target) in enumerate(more_itertools.zip_equal(rows, targetrows)):
            assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
        except:
          raise ValueError("Error in "+filename)

      with PIL.Image.open(dbloadroot/SlideID/"dbload"/f"{SlideID}_qptiff.jpg") as img, \
           PIL.Image.open(thisfolder/"data"/"reference"/"prepdb"/SlideID/f"{SlideID}_qptiff.jpg") as targetimg:
        np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))

      for log in logs:
        checkwindowsnewlines(log)
    except:
      self.saveoutput()
      raise
    finally:
      self.removeoutput()

  def testPrepDbFastUnits(self, SlideID="M21_1"):
    self.testPrepDb(SlideID, units="fast")

  def testPrepDbPolaris(self, **kwargs):
    from .data.YZ71.im3.Scan3.assembleqptiff import assembleqptiff
    assembleqptiff()
    self.testPrepDb(SlideID="YZ71", skipannotations=True, **kwargs)
  def testPrepDbPolarisFastUnits(self):
    self.testPrepDbPolaris(units="fast")

  def testPrepDbM206FastUnits(self):
    from .data.M206.im3.Scan1.assembleqptiff import assembleqptiff
    assembleqptiff()
    xmlfile = thisfolder/"data"/"M206"/"im3"/"Scan1"/"M206_Scan1.annotations.polygons.xml"
    with open(xmlfile) as f:
      contents = f.read()
    with temporarilyreplace(xmlfile, contents.replace("Good tissue", "Good tisue")):
      self.testPrepDb(SlideID="M206", units="fast_microns")
