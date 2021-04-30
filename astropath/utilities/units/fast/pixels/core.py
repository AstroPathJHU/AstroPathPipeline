import numba as nb
from ..common import micronstopixels, pixelstomicrons

def Distance(*, pscale, pixels=None, microns=None, centimeters=None, power=1, defaulttozero=False):
  if pixels is not None is microns is centimeters:
    return pixels
  elif microns is not None is pixels is centimeters:
    try:
      return micronstopixels(microns, pscale, power)
    except TypeError:
      return microns * Distance(microns=1, pscale=pscale, power=power)
  elif centimeters is not None is pixels is microns:
    microns = centimeters * (1e4**power if power and centimeters else 1)
    return Distance(pscale=pscale, microns=microns, power=power)
  else:
    raise TypeError("Have to provide exactly one of pixels, microns, or centimeters")

def pixels(distance, *, pscale, power=1):
  return distance
def microns(distance, *, pscale, power=1):
  try:
    return pixelstomicrons(distance, pscale, power)
  except TypeError:
    return distance * microns(1, pscale=pscale, power=power)

@nb.vectorize([nb.float64(nb.float64, nb.float64, nb.float64, nb.float64)])
def __convertpscale(distance, oldpscale, newpscale, power):
  return micronstopixels(pixelstomicrons(distance, oldpscale, power), newpscale, power)
def convertpscale(distance, oldpscale, newpscale, power=1):
  try:
    return __convertpscale(distance, oldpscale, newpscale, power)
  except TypeError:
    return distance * convertpscale(1, oldpscale, newpscale, power)
