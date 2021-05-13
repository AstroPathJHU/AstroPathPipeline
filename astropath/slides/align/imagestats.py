from ...utilities import units
from ...utilities.units.dataclasses import DataClassWithPscale, distancefield

class ImageStats(DataClassWithPscale):
  """
  Gives statistics about an HPF image.

  n: id of the HPF
  mean, min, max, std: the average, minimum, maximum, and standard deviation
                       of the pixel fluxes
  cx, cy: the center of the HPF in integer microns
  """

  n: int
  mean: float
  min: float
  max: float
  std: float
  cx: units.Distance = distancefield(pixelsormicrons="microns", dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons="microns", dtype=int)
