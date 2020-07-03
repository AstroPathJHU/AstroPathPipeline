import numpy as np, pathlib, unittest
from ..alignment.alignmentset import AlignmentSet
from ..badregions.dustspeck import DustSpeckFinder
from ..badregions.tissuefold import TissueFoldFinderSimple, TissueFoldFinderByCell

thisfolder = pathlib.Path(__file__).parent

class TestBadRegions(unittest.TestCase):
  savedir = thisfolder/"badregions_test_for_jenkins"

  @classmethod
  def setUpClass(cls):
    cls.images = []
    A = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    A.getDAPI(writeimstat=False)
    cls.images.append(A.rectangles[21].image)

    cls.writeoutreference = False
    try:
      with np.load(thisfolder/"reference"/"badregions"/"badregionsreference.npz") as f:
        cls.reference = dict(f)
    except IOError:
      cls.reference = {}
    cls.seenclasses = set()
    cls.savedir.mkdir(exist_ok=True)

  @classmethod
  def tearDownClass(cls):
    unknownkeys = set(cls.reference.keys()) - cls.seenclasses
    for _ in unknownkeys:
      cls.writeoutreference = True
      del cls.reference[_]
    if cls.writeoutreference:
      np.savez_compressed(cls.savedir/"badregions.npz", **cls.reference)
    if unknownkeys:
      raise ValueError(f"Unknown arrays in badregionsreference.npz: {', '.join(unknownkeys)}")

  def generaltest(self, BRFclass, imageindex, **kwargs):
    brf = BRFclass(self.images[imageindex])
    self.seenclasses.add(BRFclass.__name__)
    badregions = brf.badregions(**kwargs)
    try:
      np.testing.assert_array_equal(badregions, self.reference[BRFclass.__name__])
    except:
      self.reference[BRFclass.__name__] = badregions
      type(self).writeoutreference = True
      brf.show(saveas=self.savedir/f"badregions_{BRFclass.__name__}.pdf", **kwargs)
      raise

  def testTissueFoldFinderSimple(self):
    self.generaltest(TissueFoldFinderSimple, 0, threshold=0.15)

  def testTissueFoldFinderByCell(self):
    self.generaltest(TissueFoldFinderByCell, 0, threshold=0.15)

  def testDustSpeckFinder(self):
    self.generaltest(DustSpeckFinder, 0)
