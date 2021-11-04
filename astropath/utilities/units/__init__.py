"""
The units module provides an interface for dealing with distances, which
are sometimes saved in units of pixels and sometimes in units of microns.
It has three modes: safe, fast_pixels, and fast_microns.  (fast is an alias
for fast_pixels.)

When running in safe mode, all distances are saved as Distance objects,
and the dimensions are checked when doing math with them.  When running in
fast_pixels or fast_microns mode, distances are saved as numbers, with the
convention 1 pixel = 1 or 1 micron = 1, respectively.

All modes are designed to produce the same output results.  In practice,
safe mode is only needed for testing.  When running modules, fast_pixels
or fast_microns should usually be equivalent, but in computation-heavy
processes one or the other might be faster.

  from . import setup_context, Distance, currentmodule
  >>> def run(mode):
  ...   with setup_context(mode):
  ...     from . import Distance, pixels, microns #not needed in code, only as a hack for doctest
  ...     print("These quantities have dimension, so their value depends on the mode")
  ...     d1 = Distance(pixels=1., pscale=2); print(d1)
  ...     d2 = Distance(microns=1., pscale=2); print(d2)
  ...     print("These quantities are dimensionless, so their value is independent of the mode")
  ...     print(d2 / d1)
  ...     print(pixels(d1, pscale=2))
  ...     print(microns(d1, pscale=2))
  >>> run("safe")
  These quantities have dimension, so their value depends on the mode
  1.0 pixels
  2.0 pixels
  These quantities are dimensionless, so their value is independent of the mode
  2.0
  1.0
  0.5
  >>> run("fast_microns")
  These quantities have dimension, so their value depends on the mode
  0.5
  1.0
  These quantities are dimensionless, so their value is independent of the mode
  2.0
  1.0
  0.5
  >>> run("fast_pixels")
  These quantities have dimension, so their value depends on the mode
  1.0
  2.0
  These quantities are dimensionless, so their value is independent of the mode
  2.0
  1.0
  0.5
"""

import contextlib
from . import core, dataclasses, drawing
from .core import Distance, onemicron, onepixel, ThingWithApscale, ThingWithImscale, ThingWithPscale, ThingWithQpscale, ThingWithScale, UnitsError

def setup(mode, baseunit=None):
  """
  set the mode for distances.
  this needs to be called before you create any astropath objects
  (like Samples or Rectangles)
  """
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

def currentargs():
  return currentmode, currentbaseunit

@contextlib.contextmanager
def setup_context(mode, baseunit=None):
  """
  Call setup in a context manager, switching back to the old mode on exit.
  """
  bkp = currentargs()
  try:
    yield setup(mode, baseunit)
  finally:
    setup(*bkp)

__all__ = [
  "convertpscale", "correlated_distances", "Distance", "distances", "onemicron", "onepixel", "ThingWithApscale", "ThingWithImscale", "ThingWithPscale", "ThingWithQpscale", "ThingWithScale", "UnitsError",
  "asdimensionless", "covariance_matrix", "microns", "nominal_value", "nominal_values", "pixels", "std_dev", "std_devs",
  "dataclasses", "drawing", "fft", "linalg", "testing",
  "setup", "setup_context",
  "np", "scipy"
]
