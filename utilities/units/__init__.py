from . import dataclasses, linalg, testing

def setup(mode):
  global correlated_distances, Distance, distances, UnitsError
  global covariance_matrix, microns, nominal_value, nominal_values, pixels, pscale, std_dev, std_devs

  for _ in dataclasses, linalg, testing:
    _.__setup(mode)

  if mode == "safe":
    from .safe.core import correlated_distances, Distance, distances, UnitsError
    from .safe.core import covariance_matrix, microns, nominal_value, nominal_values, pixels, pscale, std_dev, std_devs
  elif mode == "pixels":
    from .fast.pixels.core import correlated_distances, Distance, distances, UnitsError
    from .fast.pixels.core import covariance_matrix, microns, nominal_value, nominal_values, pixels, pscale, std_dev, std_devs
  elif mode == "microns":
    from .fast.microns.core import correlated_distances, Distance, distances, UnitsError
    from .fast.microns.core import covariance_matrix, microns, nominal_value, nominal_values, pixels, pscale, std_dev, std_devs
  else:
    raise ValueError(f"Invalid mode {mode}")

setup("safe")

__all__ = [
  "correlated_distances", "Distance", "distances", "UnitsError",
  "covariance_matrix", "microns", "nominal_value", "nominal_values", "pixels", "pscale", "std_dev", "std_devs",
  "dataclasses", "linalg", "testing",
  "setup",
]
