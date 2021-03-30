import numpy as np
from uncertainties import correlated_values, covariance_matrix, nominal_value, std_dev
from uncertainties.unumpy import nominal_values, std_devs

def setup(baseunit):
  global convertpscale, Distance, distances, pixels, microns

  if baseunit == "pixels":
    from .pixels import convertpscale, Distance, microns, pixels
  elif baseunit == "microns":
    from .microns import convertpscale, Distance, microns, pixels
  else:
    raise ValueError(f"Unknown baseunit: {baseunit}")

  distances = Distance

setup("pixels")

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
  return correlated_values(distances, covariance)

def asdimensionless(distance): return distance

__all__ = [
  "convertpscale", "correlated_distances", "Distance", "distances",
  "asdimensionless", "covariance_matrix", "microns", "nominal_value", "nominal_values", "pixels", "std_dev", "std_devs",
]
