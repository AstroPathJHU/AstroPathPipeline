import numba as nb, numpy as np, uncertainties as unc

def __vectorizetypelist(nargs):
  return [
    typ(*[typ]*nargs)
    for typ in (nb.int32, nb.int64, nb.float32, nb.float64)
  ]

@nb.vectorize(__vectorizetypelist(3))
def __micronstopixels(microns, pscale, power):
  return microns * pscale**power
@nb.vectorize(__vectorizetypelist(3))
def __pixelstomicrons(pixels, pscale, power):
  return pixels / pscale**power

def __Distance(pscale, pixels, microns, centimeters, power, defaulttozero):
  if not power or pixels == 0 or microns == 0 or centimeters == 0: pscale = None
  if (pixels is not None) + (microns is not None) + (centimeters is not None) != 1:
    raise TypeError("Have to provide exactly one of pixels, microns, or centimeters")
  if centimeters is not None:
    microns = centimeters * (1e4**power if power and centimeters else 1)

  if pixels is not None:
    return pixels
  if microns is not None:
    if pscale is None or power is None or microns == 0: return microns
    return __micronstopixels(microns, pscale, power)

def Distance(*, pscale, pixels=None, microns=None, centimeters=None, power=1, defaulttozero=False):
  return __Distance(pscale, pixels, microns, centimeters, power, defaulttozero)

distances = Distance

def correlated_distances(*, pscale=None, pixels=None, microns=None, distances=None, covariance=None, power=None):
  if (pixels is not None) + (microns is not None) + (distances is not None) != 1:
    raise TypeError("Have to provide exactly one of pixels, microns, or distances")

  if distances is not None: pixels = distances
  if microns is not None: pixels = __micronstopixels(microns)

  if covariance is None: return pixels
  return unc.correlated_values(pixels, covariance)

def pixels(distance, *, pscale, power): return distance
def microns(distance, *, pscale, power):
  return __pixelstomicrons(distance, pscale, power)
def asdimensionless(distance): return distance
