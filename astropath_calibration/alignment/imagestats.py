from ..utilities.units.dataclasses import DataClassWithPscale, distancefield

class ImageStats(DataClassWithPscale):
  pixelsormicrons = "microns"

  n: int
  mean: float
  min: float
  max: float
  std: float
  cx: distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cy: distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
