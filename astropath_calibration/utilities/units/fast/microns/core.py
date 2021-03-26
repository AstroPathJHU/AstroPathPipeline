import uncertainties.unumpy as unp
from ..common import micronstopixels, pixelstomicrons

def Distance(*, pscale, pixels=None, microns=None, centimeters=None, power=1, defaulttozero=False):
  if pixels is not None is microns is centimeters:
    try:
      return pixelstomicrons(pixels, pscale, power)
    except TypeError:
      nominal = unp.nominal_values(pixels)
      return pixels * Distance(pixels=nominal, pscale=pscale, power=power) / nominal
  elif microns is not None is pixels is centimeters:
    return microns
  elif centimeters is not None is pixels is microns:
    microns = centimeters * (1e4**power if power and centimeters else 1)
    return microns
  else:
    raise TypeError("Have to provide exactly one of pixels, microns, or centimeters")

def pixels(distance, *, pscale, power=1):
  try:
    return micronstopixels(distance, pscale, power)
  except TypeError:
    nominal = unp.nominal_values(distance)
    return distance * pixels(nominal, pscale=pscale, power=power) / nominal
def microns(distance, *, pscale, power=1):
  return distance
def convertpscale(distance, oldpscale, newpscale, power=1):
  return distance
