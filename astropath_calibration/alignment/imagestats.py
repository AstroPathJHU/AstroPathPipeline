import dataclassy
from ..utilities import units
from ..utilities.units.dataclasses import DataClassWithPscaleFrozen, distancefield

@dataclassy.dataclass(frozen=True)
class ImageStats(DataClassWithPscaleFrozen):
  pixelsormicrons = "microns"

  n: int
  mean: float
  min: float
  max: float
  std: float
  cx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  readingfromfile: dataclassy.InitVar[bool] = False
