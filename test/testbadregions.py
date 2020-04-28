import numpy as np, pathlib, unittest
from ..alignment.alignmentset import AlignmentSet
from ..badregions.badregions import BadRegionFinder

thisfolder = pathlib.Path(__file__).parent

class TestBadRegions(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    A = AlignmentSet(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    A.getDAPI(writeimstat=False)
    cls.image = A.rectangles[21].image

  def testBadRegionFinder(self):
    f = BadRegionFinder(self.image)
    kwargs = {"threshold": 0.02}
    badregions = f.badregions(**kwargs)
    try:
      with np.load(thisfolder/"reference"/"badregions"/"badregionsreference.npz") as f:
        reference, = f.values()
      np.testing.assert_array_equal(badregions, reference)
    except:
      savedir = thisfolder/"badregions_test_for_jenkins"
      savedir.mkdir(exist_ok=True)
      np.savez_compressed(savedir/"badregions.npz", badregions)
      f.show(saveas=savedir/"badregions.pdf", **kwargs)
      raise
