import contextlib
from . import dataclasses, drawing, numpy as np, optimize
from .core import UnitsError

def setup(mode):
  global currentmodule
  global correlated_distances, distances
  global asdimensionless, covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
  global angle, isclose, linspace
  global currentmode, unitdtype

  for _ in dataclasses, np, optimize:
    _.__setup(mode)

  try:
    if mode == "safe":
      from . import safe as currentmodule
      from .safe import correlated_distances, distances
      from .safe import asdimensionless, covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
      from .safe import angle, isclose, linspace
      unitdtype = object
    elif mode == "fast":
      from . import fast as currentmodule
      from .fast import correlated_distances, distances
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

class Distance:
  def __new__(self, *args, **kwargs):
    return currentmodule.Distance(*args, **kwargs)

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
  "dataclasses", "drawing", "fft", "linalg", "testing",
  "angle", "isclose", "linspace",
  "setup", "setup_context",
]
