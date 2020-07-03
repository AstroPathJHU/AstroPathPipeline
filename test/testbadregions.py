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

    A = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M55_1", selectrectangles=[678])
    A.getDAPI(writeimstat=False, keeprawimages=True)
    cls.images.append(A.rectangles[0].rawimage)

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

  def generaltest(self, BRFclass, imageindex, *, expectallgood=False, **kwargs):
    brf = BRFclass(self.images[imageindex])
    badregions = brf.badregions(**kwargs)

    try:
      if expectallgood:
        reference = np.zeros_like(badregions, dtype=badregions.dtype)
      else:
        reference = self.reference[BRFclass.__name__]

        if not np.any(reference):
          raise AssertionError("Reference doesn't have any bad regions, are you sure this is right?  If it is, do expectallgood=True.")
        if BRFclass.__name__ in self.seenclasses:
          raise RuntimeError(f"Trying to run {BRFclass.__name__} on a second image.  Testing is only set up to test each class on one image, unless expectallgood")
        self.seenclasses.add(BRFclass.__name__)

      np.testing.assert_array_equal(badregions, reference)

    except:
      if not expectallgood:
        self.reference[BRFclass.__name__] = badregions
        type(self).writeoutreference = True
        self.seenclasses.add(BRFclass.__name__)
      brf.show(saveas=self.savedir/f"badregions_{BRFclass.__name__}.pdf", **kwargs)
      raise

  def testTissueFoldFinderSimple(self):
    self.generaltest(TissueFoldFinderSimple, 0, threshold=0.15)

  def testTissueFoldFinderByCell(self):
    self.generaltest(TissueFoldFinderByCell, 0, threshold=0.15)

  def testDustSpeckFinderNoSpeck(self):
    self.generaltest(DustSpeckFinder, 0, expectallgood=True)

  def testDustSpeckFinderWithSpeck(self):
    self.generaltest(DustSpeckFinder, 1)
