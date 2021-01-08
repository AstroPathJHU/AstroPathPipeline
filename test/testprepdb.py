import more_itertools, numpy as np, os, pathlib, PIL.Image, re, unittest
from astropath_calibration.baseclasses.csvclasses import Annotation, Batch, Constant, Globals, Polygon, QPTiffCsv, Region, Vertex
from astropath_calibration.baseclasses.sample import SampleDef
from astropath_calibration.baseclasses.overlap import Overlap, rectangleoverlaplist_fromcsvs
from astropath_calibration.baseclasses.rectangle import Rectangle
from astropath_calibration.prepdb.prepdbsample import PrepdbSample
from astropath_calibration.utilities import units
from astropath_calibration.utilities.misc import re_subs
from astropath_calibration.utilities.tableio import readtable
from .testbase import assertAlmostEqual

thisfolder = pathlib.Path(__file__).parent

class TestPrepDb(unittest.TestCase):
  def setUp(self):
    self.maxDiff = None

  def testPrepDb(self, SlideID="M21_1"):
    logs = (
      thisfolder/"data"/"logfiles"/"prepdb.log",
      thisfolder/"data"/SlideID/"logfiles"/f"{SlideID}-prepdb.log",
    )
    for log in logs:
      try:
        log.unlink()
      except FileNotFoundError:
        pass

    samp = SampleDef(SlideID=SlideID, SampleID=0, Project=0, Cohort=0, root=thisfolder/"data")
    sample = PrepdbSample(thisfolder/"data", samp, uselogfiles=True, xmlfolders=[thisfolder/"data"/"raw"/SlideID])
    with sample:
      sample.writemetadata()

    for filename, cls, extrakwargs in (
      (f"{SlideID}_annotations.csv", Annotation, {}),
      (f"{SlideID}_batch.csv", Batch, {}),
      (f"{SlideID}_constants.csv", Constant, {"pscale": sample.pscale, "apscale": sample.apscale, "qpscale": sample.qpscale, "readingfromfile": True}),
      (f"{SlideID}_globals.csv", Globals, {"pscale": sample.pscale}),
      (f"{SlideID}_overlap.csv", Overlap, {"pscale": sample.pscale, "nclip": sample.nclip, "rectangles": sample.rectangles}),
      (f"{SlideID}_qptiff.csv", QPTiffCsv, {"pscale": sample.pscale}),
      (f"{SlideID}_rect.csv", Rectangle, {"pscale": sample.pscale}),
      (f"{SlideID}_vertices.csv", Vertex, {"apscale": sample.apscale}),
      (f"{SlideID}_regions.csv", Region, {"apscale": sample.apscale, "pscale": sample.pscale}),
    ):
      if filename == "M21_1_globals.csv": continue
      try:
        rows = readtable(thisfolder/"data"/SlideID/"dbload"/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        targetrows = readtable(thisfolder/"reference"/"prepdb"/SlideID/filename, cls, extrakwargs=extrakwargs, checkorder=True)
        for i, (row, target) in enumerate(more_itertools.zip_equal(rows, targetrows)):
          assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
      except:
        raise ValueError("Error in "+filename)

    with PIL.Image.open(thisfolder/"data"/SlideID/"dbload"/f"{SlideID}_qptiff.jpg") as img, \
         PIL.Image.open(thisfolder/"reference"/"prepdb"/SlideID/f"{SlideID}_qptiff.jpg") as targetimg:
      np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))

      for log in logs:
        ref = thisfolder/"reference"/"prepdb"/SlideID/log.name
        with open(ref) as fref, open(log) as fnew:
          subs = (";[^;]*$", ""), (r"(WARNING: (component tiff|xml files|constants\.csv)).*$", r"\1")
          from astropath_calibration.utilities.version import astropathversion
          refsubs = *subs, (r"(prepdb )v[\w+.]+", rf"\1{astropathversion}")
          newsubs = *subs,
          refcontents = os.linesep.join([re_subs(line, *refsubs, flags=re.MULTILINE) for line in fref.read().splitlines() if "Biggest time difference" not in line])+os.linesep
          newcontents = os.linesep.join([re_subs(line, *newsubs, flags=re.MULTILINE) for line in fnew.read().splitlines() if "Biggest time difference" not in line])+os.linesep
          self.assertEqual(newcontents, refcontents)

  def testPrepDbFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testPrepDb(SlideID)

  def testPrepDbPolaris(self):
    from .data.YZ71.im3.Scan3.assembleqptiff import assembleqptiff
    assembleqptiff()
    self.testPrepDb(SlideID="YZ71")
  def testPrepDbPolarisFastUnits(self):
    self.testPrepDbFastUnits(SlideID="YZ71")

  def testPrepDbM206FastUnits(self):
    from .data.M206.im3.Scan1.assembleqptiff import assembleqptiff
    assembleqptiff()
    self.testPrepDbFastUnits(SlideID="M206")

  def testRectangleOverlapList(self):
    l = rectangleoverlaplist_fromcsvs(thisfolder/"data"/"M21_1"/"dbload", layer=1)
    islands = l.islands()
    self.assertEqual(len(islands), 2)
    l2 = rectangleoverlaplist_fromcsvs(thisfolder/"data"/"M21_1"/"dbload", selectrectangles=lambda x: x.n in islands[0], layer=1)
    self.assertEqual(l2.islands(), [l.islands()[0]])

  def testRectangleOverlapListFastUnits(self):
    with units.setup_context("fast"):
      self.testRectangleOverlapList()

  def testPolygonAreas(self):
    p = Polygon(pixels="POLYGON ((1 1,2 1,2 2,1 2,1 1))", pscale=5, apscale=3)
    assertAlmostEqual(units.convertpscale(p.totalarea, p.apscale, p.pscale, power=2), p.onepixel**2, rtol=1e-15)
    p = Polygon(pixels="POLYGON ((1 1,4 1,4 4,1 4,1 1),(2 2,2 3,3 3,3 2,2 2))", pscale=5, apscale=3)
    assertAlmostEqual(units.convertpscale(p.totalarea, p.apscale, p.pscale, power=2), 8*p.onepixel**2, rtol=1e-15)

    try:
      xysx2 = units.distances(pixels=np.random.randint(-10, 11, (2, 100, 2)), pscale=3)
      vertices = [[Vertex(regionid=None, vid=i, x=x, y=y, pscale=5, apscale=3) for i, (x, y) in enumerate(xys) if x or y] for xys in xysx2]
      p1, p2 = [Polygon(vertices=[vv]) for vv in vertices]
      assertAlmostEqual(p1.totalarea+p2.totalarea, (p1+p2).totalarea)
      assertAlmostEqual(p1.totalarea-p2.totalarea, (p1-p2).totalarea)
    except:
      print(xysx2)
      raise

  def testPolygonAreasFastUnits(self):
    with units.setup_context("fast"):
      self.testPolygonAreas()
