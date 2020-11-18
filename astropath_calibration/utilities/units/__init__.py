import contextlib
from . import dataclasses, drawing
from .core import convertpscale, UnitsError

def setup(mode):
  global currentmodule
  global correlated_distances, distances
  global asdimensionless, covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
  global currentmode, unitdtype
  global np, scipy

  for _ in dataclasses,:
    _.__setup(mode)

  try:
    if mode == "safe":
      from . import safe as currentmodule
      from .safe import correlated_distances, distances
      from .safe import asdimensionless, covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
      from .safe import numpy as np, scipy
      unitdtype = object
    elif mode == "fast":
      from . import fast as currentmodule
      from .fast import correlated_distances, distances
      from .fast import asdimensionless, covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
      import numpy as np, scipy
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
  "convertpscale", "correlated_distances", "Distance", "distances", "UnitsError",
  "asdimensionless", "covariance_matrix", "microns", "nominal_value", "nominal_values", "pixels", "std_dev", "std_devs",
  "dataclasses", "drawing", "fft", "linalg", "testing",
  "setup", "setup_context",
  "np", "scipy"
]