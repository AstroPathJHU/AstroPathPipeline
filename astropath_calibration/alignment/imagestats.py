import dataclasses
from ..utilities import units
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield, pscalefield

@dataclasses.dataclass(frozen=True)
class ImageStats(DataClassWithDistances):
  pixelsormicrons = "microns"

  n: int
  mean: float
  min: float
  max: float
  std: float
  cx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  pscale: float = pscalefield()
  readingfromfile: dataclasses.InitVar[bool] = False
