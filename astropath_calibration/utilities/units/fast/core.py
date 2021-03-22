import numba as nb, numpy as np, uncertainties as unc

__types = nb.int32, nb.int64, nb.float32, nb.float64
__vectorizetypelist = [
  nb.float64(nb.float64, nb.float64, nb.float64),
]

@nb.vectorize(__vectorizetypelist)
def __micronstopixels(microns, pscale, power):
  return microns * pscale**power
@nb.vectorize(__vectorizetypelist)
def __pixelstomicrons(pixels, pscale, power):
  return pixels / pscale**power

__vectorizetypelist = [
  *[
    typ(typ, typ, typ, nb.boolean)
    for typ in __types
  ], *[
    nb.float64(nb.float64, typ, typ, nb.boolean)
    for typ in __types
  ],
  nb.float64(nb.float64, nb.int32, nb.int64, nb.boolean),
]

def Distance(*, pscale, pixels=None, microns=None, centimeters=None, power=1, defaulttozero=False):
  if pixels is not None is microns is centimeters:
    return pixels
  elif microns is not None is pixels is centimeters:
    return __micronstopixels(microns, pscale, power)
  elif centimeters is not None is pixels is microns:
    microns = centimeters * (1e4**power if power and centimeters else 1)
    return __micronstopixels(microns, pscale, power)
  else:
    raise TypeError("Have to provide exactly one of pixels, microns, or centimeters")

distances = Distance

def correlated_distances(*, pscale=None, pixels=None, microns=None, distances=None, covariance=None, power=None):
  if (pixels is not None) + (microns is not None) + (distances is not None) != 1:
    raise TypeError("Have to provide exactly one of pixels, microns, or distances")

  if distances is not None: pixels = distances
  if microns is not None: pixels = __micronstopixels(microns)

  if covariance is None: return pixels
  return unc.correlated_values(pixels, covariance)

def pixels(distance, *, pscale, power=1): return distance
def microns(distance, *, pscale, power=1):
  return __pixelstomicrons(distance, pscale, power)
def asdimensionless(distance): return distance
