import hashlib, numpy as np, os, pathlib, unittest
from astropath_calibration.baseclasses.csvclasses import Vertex
from astropath_calibration.baseclasses.polygon import Polygon
from astropath_calibration.baseclasses.overlap import rectangleoverlaplist_fromcsvs
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
    assertAlmostEqual(units.convertpscale(p.totalarea, p.apscale, p.pscale, power=2), p.onepixel**2, rtol=1e-15)
    p = Polygon(pixels="POLYGON ((1 1,4 1,4 4,1 4,1 1),(2 2,2 3,3 3,3 2,2 2))", pscale=5, apscale=3)
    assertAlmostEqual(units.convertpscale(p.totalarea, p.apscale, p.pscale, power=2), 8*p.onepixel**2, rtol=1e-15)

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
      assertAlmostEqual(p1.totalarea-p2.totalarea, (p1-p2).totalarea)
      assertAlmostEqual((p1-p1).totalarea, 0)
      for p in p1, p2, p1-p2:
        assertAlmostEqual(
          units.convertpscale(p.totalarea, p.apscale, p.pscale, power=2),
          p.gdalpolygon().Area() * p.onepixel**2
        )
    except:
      print(xysx2)
      raise

  def testPolygonAreasFastUnits(self):
    with units.setup_context("fast"):
      self.testPolygonAreas()
