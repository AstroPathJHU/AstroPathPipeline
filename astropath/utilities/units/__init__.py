import contextlib
from . import core, dataclasses, drawing
from .core import Distance, onemicron, onepixel, ThingWithApscale, ThingWithImscale, ThingWithPscale, ThingWithQpscale, ThingWithZoomedscale, UnitsError

def setup(mode, baseunit=None):
  if "_" in mode and baseunit is None:
    return setup(*mode.split("_"))

  global convertpscale, currentmodule, currentbaseunit
  global correlated_distances, distances
  global asdimensionless, covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
  global currentmode, unitdtype
  global np, scipy

  try:
    if mode == "safe":
      if baseunit is not None: raise ValueError("Provided baseunit for safe units")
      from . import safe as currentmodule
      from .safe import convertpscale, correlated_distances, distances
      from .safe import asdimensionless, covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
      from .safe import numpy as np, scipy
      unitdtype = object
    elif mode == "fast":
      from . import fast as currentmodule
      if baseunit is None: baseunit = "pixels"
      currentmodule.setup(baseunit)
      from .fast import convertpscale, correlated_distances, distances
      from .fast import asdimensionless, covariance_matrix, microns, nominal_value, nominal_values, pixels, std_dev, std_devs
      import numpy as np, scipy
      unitdtype = float
    else:
      raise ValueError(f"Invalid mode {mode}")
    core.currentmodule = currentmodule
  except:
    currentmode = currentbaseunit = None
    raise
  else:
    currentmode = mode
    currentbaseunit = baseunit

  for _ in dataclasses,:
    _.__setup(mode)

setup("safe")

@contextlib.contextmanager
def setup_context(mode, baseunit="pixels"):
  bkp = currentmode, currentbaseunit
  try:
    yield setup(mode)
  finally:
    setup(*bkp)

__all__ = [
  "convertpscale", "correlated_distances", "Distance", "distances", "onemicron", "onepixel", "ThingWithApscale", "ThingWithImscale", "ThingWithPscale", "ThingWithQpscale", "ThingWithZoomedscale", "UnitsError",
  "asdimensionless", "covariance_matrix", "microns", "nominal_value", "nominal_values", "pixels", "std_dev", "std_devs",
  "dataclasses", "drawing", "fft", "linalg", "testing",
  "setup", "setup_context",
  "np", "scipy"
]
