import contextlib
from . import dataclasses, linalg, testing
from .core import UnitsError

def setup(mode):
  global correlated_distances, Distance, distances
  global covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
  global currentmode, unitdtype

  for _ in dataclasses, linalg, testing:
    _.__setup(mode)

  try:
    if mode == "safe":
      from .safe import correlated_distances, Distance, distances
      from .safe import covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
      unitdtype = object
    elif mode == "fast":
      from .fast import correlated_distances, Distance, distances
      from .fast import covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
      unitdtype = float
    else:
      raise ValueError(f"Invalid mode {mode}")
  except:
    currentmode = None
    raise
  else:
    currentmode = mode

setup("safe")

@contextlib.contextmanager
def setup_context(mode):
  bkp = currentmode
  try:
    yield setup(mode)
  finally:
    setup(bkp)

__all__ = [
  "correlated_distances", "Distance", "distances", "UnitsError",
  "covariance_matrix", "microns", "nominal_value", "nominal_values", "pixels", "std_dev", "std_devs",
  "dataclasses", "linalg", "testing",
  "setup",
]
