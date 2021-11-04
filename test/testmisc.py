import cv2, datetime, hashlib, more_itertools, numpy as np, os, pathlib, skimage
from astropath.shared.annotationpolygonxmlreader import writeannotationcsvs
from astropath.shared.contours import findcontoursaspolygons
from astropath.shared.csvclasses import Annotation, Region, Vertex
from astropath.shared.overlap import rectangleoverlaplist_fromcsvs
from astropath.shared.polygon import Polygon, PolygonFromGdal, SimplePolygon
from astropath.shared.rectangle import Rectangle
from astropath.slides.prepdb.prepdbsample import PrepDbSample
from astropath.shared.samplemetadata import APIDDef, MakeSampleDef, SampleDef
from astropath.utilities import units
from astropath.utilities.logger import printlogger
from astropath.utilities.tableio import readtable, writetable
from .testbase import assertAlmostEqual, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestMisc(TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    return [
      thisfolder/"test_for_jenkins"/"misc"/"M206_annotations.csv",
      thisfolder/"test_for_jenkins"/"misc"/"M206_regions.csv",
      thisfolder/"test_for_jenkins"/"misc"/"M206_vertices.csv",
      thisfolder/"test_for_jenkins"/"misc"/"M21_1_annotations.csv",
      thisfolder/"test_for_jenkins"/"misc"/"M21_1_regions.csv",
      thisfolder/"test_for_jenkins"/"misc"/"M21_1_vertices.csv",
      thisfolder/"test_for_jenkins"/"misc"/"makesampledef"/"sampledef.csv",
      thisfolder/"test_for_jenkins"/"misc"/"tableappend"/"sampledef.csv",
      thisfolder/"test_for_jenkins"/"misc"/"tableappend"/"noheader.csv",
    ]
  def testRectangleOverlapList(self):
    l = rectangleoverlaplist_fromcsvs(thisfolder/"data"/"M21_1"/"dbload", layer=1)
    islands = l.islands()
    self.assertEqual(len(islands), 2)
    l2 = rectangleoverlaplist_fromcsvs(thisfolder/"data"/"M21_1"/"dbload", selectrectangles=lambda x: x.n in islands[0], layer=1)
    self.assertEqual(l2.islands(), [l.islands()[0]])

  def testRectangleOverlapListFastUnits(self):
    with units.setup_context("fast"):
      self.testRectangleOverlapList()

  def testPolygonAreas(self, seed=None):
    logger = printlogger("polygonareas")
    p = PolygonFromGdal(pixels="POLYGON ((1 1,2 1,2 2,1 2,1 1))", pscale=5, apscale=3)
    assertAlmostEqual(p.area, p.onepixel**2, rtol=1e-15)
    assertAlmostEqual(p.perimeter, 4*p.onepixel, rtol=1e-15)
    p = PolygonFromGdal(pixels="POLYGON ((1 1,4 1,4 4,1 4,1 1),(2 2,2 3,3 3,3 2,2 2))", pscale=5, apscale=3)
    assertAlmostEqual(p.area, 8*p.onepixel**2, rtol=1e-15)
    assertAlmostEqual(p.perimeter, 16*p.onepixel, rtol=1e-15)

    if seed is None:
      try:
        seed = int(hashlib.sha1(os.environ["BUILD_TAG"].encode("utf-8")).hexdigest(), 16)
      except KeyError:
        pass
    rng = np.random.default_rng(seed)
    try:
      areas = 0
      while np.any(areas == 0):
        xysx2 = units.distances(pixels=rng.integers(-10, 11, (2, 100, 2)), pscale=3)
        vertices = [[Vertex(regionid=None, vid=i, x=x, y=y, pscale=5, apscale=3) for i, (x, y) in enumerate(xys) if x or y] for xys in xysx2]
        p1, p2 = [SimplePolygon(vertices=vv) for vv in vertices]
        areas = np.array([p1.area, p2.area, (p1-p2).area])
      assertAlmostEqual(p1.area-p2.area, (p1-p2).area)
      assertAlmostEqual((p1-p1).area, 0)
      for p in p1, p2, p1-p2:
        assertAlmostEqual(
          p.area,
          p.gdalpolygon().Area() * p.onepixel**2,
        )
        assertAlmostEqual(
          p.perimeter,
          p.gdalpolygon().Boundary().Length() * p.onepixel,
        )
        self.maxDiff = None
        self.assertEqual(
          str(p),
          str(p.gdalpolygon(round=True)),
        )
    except:
      logger.error(xysx2)
      raise

  def testPolygonAreasFastUnitsPixels(self):
    with units.setup_context("fast_pixels"):
      self.testPolygonAreas()

  def testPolygonAreasFastUnitsMicrons(self):
    with units.setup_context("fast_microns"):
      self.testPolygonAreas()

  def testPolygonNumpyArray(self):
    fraction = ".9999" if skimage.__version__ >= "0.18" else ".0001"
    polystring = f"POLYGON((1.0001 1.0001, 1.0001 8{fraction}, 8{fraction} 8{fraction}, 8{fraction} 1.0001, 1.0001 1.0001), (4.0001 5{fraction}, 7{fraction} 5{fraction}, 7{fraction} 4.0001, 4.0001 4.0001))"
    poly = PolygonFromGdal(pixels=polystring, pscale=1, apscale=3)
    nparray = poly.numpyarray(shape=(10, 10), dtype=np.uint8)
    #doesn't work for arbitrary polygons unless you increase the tolerance, but works for a polygon with right angles
    assertAlmostEqual(poly.area / poly.onepixel**2, np.sum(nparray), rtol=1e-3)

    poly2, = findcontoursaspolygons(nparray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=poly.pscale, apscale=poly.apscale)
    #does not equal poly1, some gets eaten away

  def testComplicatedPolygon(self):
    polystring1 = "POLYGON((1 1, 1 9, 9 9, 9 1, 1 1))"
    polystring2 = "POLYGON((2 2, 2 8, 8 8, 8 2, 2 2))"
    polystring3 = "POLYGON((3 3, 3 7, 7 7, 7 3, 3 3))"
    poly1 = PolygonFromGdal(pixels=polystring1, pscale=1, apscale=3)
    poly2 = PolygonFromGdal(pixels=polystring2, pscale=1, apscale=3)
    poly3 = PolygonFromGdal(pixels=polystring3, pscale=1, apscale=3)
    inner = Polygon(poly2, [poly3])
    poly = Polygon(poly1, [inner])
    assertAlmostEqual(poly.area, 44*poly.onepixel**2)
    assertAlmostEqual(poly.perimeter, 72*poly.onepixel)

  def testPolygonNumpyArrayFastUnits(self):
    with units.setup_context("fast_microns"):
      self.testPolygonNumpyArray()

  def testStandaloneAnnotations(self, SlideID="M21_1"):
    try:
      folder = thisfolder/"test_for_jenkins"/"misc"
      s = PrepDbSample(thisfolder/"data", SlideID)
      writeannotationcsvs(folder, s.annotationspolygonsxmlfile, csvprefix=SlideID)
      for filename, cls in (
        (f"{SlideID}_annotations.csv", Annotation),
        (f"{SlideID}_vertices.csv", Vertex),
        (f"{SlideID}_regions.csv", Region),
      ):
        try:
          rows = s.readtable(folder/filename, cls, checkorder=True, checknewlines=True)
          targetrows = s.readtable(thisfolder/"data"/"reference"/"prepdb"/SlideID/filename, cls, checkorder=True, checknewlines=True)
          for i, (row, target) in enumerate(more_itertools.zip_equal(rows, targetrows)):
            assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
        except:
          raise ValueError("Error in "+filename)
    except:
      self.saveoutput()
      raise
    finally:
      self.removeoutput()

  def testStandaloneAnnotationsFastUnits(self, SlideID="M206"):
    with units.setup_context("fast"):
      self.testStandaloneAnnotations(SlideID=SlideID)

  def testPolygonVertices(self):
    polystring = "POLYGON ((1 1,2 1,2 2,1 2,1 1))"
    p = PolygonFromGdal(pixels=polystring, pscale=5, apscale=3)
    p2 = SimplePolygon(vertices=p.outerpolygon.vertices)
    self.assertEqual(str(p), polystring)
    self.assertEqual(str(p2), polystring)
    assertAlmostEqual(p.outerpolygon.vertices, p2.vertices)
    assertAlmostEqual(p.outerpolygon.vertexarray, p2.vertexarray)

  def testPolygonVerticesFastUnits(self):
    with units.setup_context("fast_microns"):
      self.testPolygonVertices()

  def testSampleDef(self):
    self.maxDiff = None
    s1 = SampleDef(samp="M21_1", root=thisfolder/"data")
    s2 = SampleDef(samp="M21_1", apidfile=thisfolder/"data"/"AstropathAPIDdef.csv", Scan=s1.Scan, SampleID=s1.SampleID)
    s3 = SampleDef(samp="M21_1", apidfile=thisfolder/"data"/"AstropathAPIDdef.csv", root=thisfolder/"data")
    APID, = {APID for APID in readtable(thisfolder/"data"/"AstropathAPIDdef.csv", APIDDef) if APID.SlideID == "M21_1"}
    s4 = SampleDef(samp=APID, Scan=s1.Scan, SampleID=s1.SampleID)
    s5 = SampleDef(samp=APID, root=thisfolder/"data")
    self.assertEqual(s1, s2)
    self.assertEqual(s1, s3)
    self.assertEqual(s1, s4)
    self.assertEqual(s1, s5)

  def testMakeSampleDef(self):
    self.maxDiff = None
    outfile = thisfolder/"test_for_jenkins"/"misc"/"makesampledef"/"sampledef.csv"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    reference = thisfolder/"data"/"sampledef.csv"
    args = [os.fspath(thisfolder/"data"), "--apidfile", os.fspath(thisfolder/"data"/"AstropathAPIDdef.csv"), "--first-sample-id", "1", "--outfile", os.fspath(outfile)]
    MakeSampleDef.runfromargumentparser(args)

    try:
      rows = readtable(outfile, SampleDef, checkorder=True, checknewlines=True)
      targetrows = readtable(reference, SampleDef, checkorder=True, checknewlines=True)

      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testTableAppend(self):
    self.maxDiff = None
    outfile = thisfolder/"test_for_jenkins"/"misc"/"tableappend"/"sampledef.csv"
    noheader = outfile.parent/"noheader.csv"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    reference = thisfolder/"data"/"sampledef.csv"
    sampledefs = readtable(reference, SampleDef, checkorder=True, checknewlines=True)
    firstbatch = sampledefs[:3]
    secondbatch = sampledefs[3:]
    writetable(outfile, firstbatch)
    writetable(outfile, secondbatch, append=True)
    writetable(noheader, firstbatch, header=False)
    with self.assertRaises(ValueError):
      writetable(noheader, secondbatch, append=True)
    onepixel = units.Distance(pixels=1, pscale=1)
    with self.assertRaises(ValueError):
      writetable(noheader, [Vertex(regionid=1, vid=1, x=onepixel, y=onepixel)], append=True, header=False)
    with self.assertRaises(ValueError):
      writetable(noheader, [Rectangle(pscale=1, n=1, x=onepixel, y=onepixel, cx=onepixel, cy=onepixel, w=onepixel, h=onepixel, file='file.im3', t=datetime.datetime.now())], append=True, header=False)
    writetable(noheader, secondbatch, append=True, header=False)

    try:
      for rows in (
        readtable(outfile, SampleDef, checkorder=True, checknewlines=True),
        readtable(noheader, SampleDef, checkorder=True, checknewlines=True, header=False),
      ):
        for row, target in more_itertools.zip_equal(rows, sampledefs):
          assertAlmostEqual(row, target)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()
