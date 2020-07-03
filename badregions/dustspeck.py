from .badregions import BadRegionFinderRegionMean, BadRegionFinderRegionStd

class DustSpeckFinder(BadRegionFinderRegionMean, BadRegionFinderRegionStd):
  def __init__(self, image, *, minradius=200):
    super().__init__(image, blocksize=minradius//2, blockoffset=minradius//4)

  def badregions(self, *, threshold=0.5):
    stdovermean = self.std / self.mean
    iteration1 = stdovermean < threshold
    return iteration1
