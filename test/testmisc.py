import cv2, hashlib, numpy as np, os, pathlib, unittest
from astropath_calibration.baseclasses.csvclasses import Vertex
from astropath_calibration.baseclasses.polygon import Polygon
from astropath_calibration.baseclasses.overlap import rectangleoverlaplist_fromcsvs
from astropath_calibration.geom.contours import findcontoursaspolygons
from astropath_calibration.utilities import units
from .testbase import assertAlmostEqual

thisfolder = pathlib.Path(__file__).parent

class TestMisc(unittest.TestCase):
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
    p = Polygon(pixels="POLYGON ((1 1,2 1,2 2,1 2,1 1))", pscale=5, apscale=3)
    assertAlmostEqual(p.area, p.onepixel**2, rtol=1e-15)
    assertAlmostEqual(p.perimeter, 4*p.onepixel, rtol=1e-15)
    p = Polygon(pixels="POLYGON ((1 1,4 1,4 4,1 4,1 1),(2 2,2 3,3 3,3 2,2 2))", pscale=5, apscale=3)
    assertAlmostEqual(p.area, 8*p.onepixel**2, rtol=1e-15)
    assertAlmostEqual(p.perimeter, 16*p.onepixel, rtol=1e-15)

    if seed is None:
      try:
        seed = int(hashlib.sha1(os.environ["BUILD_TAG"].encode("utf-8")).hexdigest(), 16)
      except KeyError:
        pass
    rng = np.random.default_rng(seed)
    xysx2 = units.distances(pixels=rng.integers(-10, 11, (2, 100, 2)), pscale=3)
    try:
      vertices = [[Vertex(regionid=None, vid=i, x=x, y=y, pscale=5, apscale=3) for i, (x, y) in enumerate(xys) if x or y] for xys in xysx2]
      p1, p2 = [Polygon(vertices=[vv]) for vv in vertices]
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
    except:
      print(xysx2)
      raise

  def testPolygonAreasFastUnits(self):
    with units.setup_context("fast"):
      self.testPolygonAreas()

  def testPolygonNumpyArray(self):
    polystring = "POLYGON((1 1, 1 9, 2 9, 2 2, 3 2, 3 9, 9 9, 9 1, 1 1), (4 6, 8 6, 8 4, 4 4))"
    poly = Polygon(pixels=polystring, pscale=1, apscale=3)
    nparray = poly.numpyarray(shape=(10, 10), dtype=np.uint8)
    #doesn't work for arbitrary polygons unless you increase the tolerance, but works for a polygon with right angles
    assertAlmostEqual(poly.area / poly.onepixel**2, np.sum(nparray))

    poly2, = findcontoursaspolygons(nparray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=poly.pscale, apscale=poly.apscale)
    #does not equal poly1, some gets eaten away

  def testPolygonNumpyArrayFastUnits(self):
    with units.setup_context("fast"):
      self.testPolygonNumpyArray()
