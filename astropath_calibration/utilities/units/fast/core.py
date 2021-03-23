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
    return __pixelstomicrons(pixels, pscale, power)
  elif microns is not None is pixels is centimeters:
    return microns
  elif centimeters is not None is pixels is microns:
    microns = centimeters * (1e4**power if power and centimeters else 1)
    return microns
  else:
    raise TypeError("Have to provide exactly one of pixels, microns, or centimeters")

distances = Distance

def correlated_distances(*, pscale=None, pixels=None, microns=None, distances=None, covariance=None, power=1):
  if distances is not None is pixels is microns:
    distances = distances
    one = 1
  elif pixels is not None is distances is microns:
    one = Distance(pscale=pscale, pixels=1)
    distances = np.array(pixels)*one
  elif microns is not None is distances is pixels:
    one = Distance(pscale=pscale, microns=1)
    distances = np.array(microns)*one
  else:
    raise TypeError("Have to provide exactly one of pixels, microns, or distances")

  if covariance is None:
    return distances

  covariance = covariance*one**2
  return unc.correlated_values(distances, covariance)

def pixels(distance, *, pscale, power=1):
  return __micronstopixels(distance, pscale, power)
def microns(distance, *, pscale, power=1):
  return distance
def asdimensionless(distance): return distance
