import numpy as np, uncertainties as unc

@np.vectorize
def __micronstopixels(*, microns, pscale, power):
  return microns * (pscale**power if power and microns else 1)  
@np.vectorize
def __pixelstomicrons(*, pixels, pscale, power):
  return pixels / (pscale**power if power and pixels else 1)  

def Distance(*, pscale, pixels=None, microns=None, power=1, defaulttozero=False):
  if not power or pixels == 0 or microns == 0: pscale = None
  if (pixels is not None) == (microns is not None):
    raise TypeError("Have to provide exactly one of pixels or microns")
  if pixels is not None:
    return pixels
  if microns is not None:
    return __micronstopixels(microns=microns, pscale=pscale, power=power)

distances = np.vectorize(Distance, excluded=["pscale"])

def correlated_distances(*, pscale=None, pixels=None, microns=None, distances=None, covariance=None, power=None):
  if (pixels is not None) + (microns is not None) + (distances is not None) != 1:
    raise TypeError("Have to provide exactly one of pixels, microns, or distances")

  if distances is not None: pixels = distances
  if microns is not None: pixels = __micronstopixels(microns)

  if covariance is None: return pixels
  return unc.correlated_values(pixels, covariance)

@np.vectorize
def pixels(distance, *, pscale=None, power=1): return distance
@np.vectorize
def microns(distance, *, pscale=None, power=1): return __pixelstomicrons(pixels=distance, pscale=pscale, power=power)
@np.vectorize
def asdimensionless(distance): return distance

def power(distance): return None
def pscale(distance): return None
