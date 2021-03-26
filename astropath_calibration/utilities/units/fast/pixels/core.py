import numba as nb
from uncertainties.core import AffineScalarFunc
from ..common import micronstopixels, pixelstomicrons

def Distance(*, pscale, pixels=None, microns=None, centimeters=None, power=1, defaulttozero=False):
  if pixels is not None is microns is centimeters:
    return pixels
  elif microns is not None is pixels is centimeters:
    if isinstance(microns, AffineScalarFunc):
      nominal = microns.n
      return microns * Distance(microns=nominal, pscale=pscale, power=power) / nominal
    return micronstopixels(microns, pscale, power)
  elif centimeters is not None is pixels is microns:
    microns = centimeters * (1e4**power if power and centimeters else 1)
    return Distance(pscale=pscale, microns=microns, power=power)
  else:
    raise TypeError("Have to provide exactly one of pixels, microns, or centimeters")

def pixels(distance, *, pscale, power=1):
  return distance
def microns(distance, *, pscale, power=1):
  if isinstance(distance, AffineScalarFunc):
    nominal = distance.n
    return distance * microns(nominal, pscale=pscale, power=power) / nominal
  return pixelstomicrons(distance, pscale, power)

@nb.vectorize([nb.float64(nb.float64, nb.float64, nb.float64, nb.float64)])
def __convertpscale(distance, oldpscale, newpscale, power):
  return micronstopixels(pixelstomicrons(distance, oldpscale, power), newpscale, power)
def convertpscale(distance, oldpscale, newpscale, power=1):
  return __convertpscale(distance, oldpscale, newpscale, power)
