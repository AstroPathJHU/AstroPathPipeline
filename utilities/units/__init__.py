import contextlib
from . import dataclasses, drawing, fft, linalg, optimize, testing
from .core import UnitsError

def setup(mode):
  global correlated_distances, Distance, distances
  global asdimensionless, covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
  global angle, isclose, linspace
  global currentmode, unitdtype

  for _ in dataclasses, fft, linalg, optimize, testing:
    _.__setup(mode)

  try:
    if mode == "safe":
      from .safe import correlated_distances, Distance, distances
      from .safe import asdimensionless, covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
      from .safe import angle, isclose, linspace
      unitdtype = object
    elif mode == "fast":
      from .fast import correlated_distances, Distance, distances
      from .fast import asdimensionless, covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
      from .fast import angle, isclose, linspace
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
  "asdimensionless", "covariance_matrix", "microns", "nominal_value", "nominal_values", "pixels", "std_dev", "std_devs",
  "dataclasses", "fft", "linalg", "testing",
  "angle", "isclose", "linspace",
  "setup", "setup_context",
]
