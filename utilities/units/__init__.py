import contextlib
from . import dataclasses, linalg, testing
from .core import UnitsError

def setup(mode):
  global correlated_distances, Distance, distances
  global covariance_matrix, microns, nominal_value, nominal_values, pixels, pscale, std_dev, std_devs
  global currentmode

  for _ in dataclasses, linalg, testing:
    _.__setup(mode)

  try:
    if mode == "safe":
      from .safe.core import correlated_distances, Distance, distances
      from .safe.core import covariance_matrix, microns, nominal_value, nominal_values, pixels, pscale, std_dev, std_devs
    elif mode == "fast":
      from .fast.core import correlated_distances, Distance, distances
      from .fast.core import covariance_matrix, microns, nominal_value, nominal_values, pixels, pscale, std_dev, std_devs
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
  "covariance_matrix", "microns", "nominal_value", "nominal_values", "pixels", "pscale", "std_dev", "std_devs",
  "dataclasses", "linalg", "testing",
  "setup",
]
