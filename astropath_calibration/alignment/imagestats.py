import dataclasses
from ..utilities import units
from ..utilities.units.dataclasses import DataClassWithPscale, distancefield, pscalefield

@dataclasses.dataclass(frozen=True)
class ImageStats(DataClassWithPscale):
  pixelsormicrons = "microns"

  n: int
  mean: float
  min: float
  max: float
  std: float
  cx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  readingfromfile: dataclasses.InitVar[bool] = False
