import numpy as np
from .core import Distance, pixels, power, pscale

def assert_allclose(distance1, distance2, *args, **kwargs):
  if not distance1 and not distance2: return
  try:
    np.testing.assert_array_equal(power(distance1), power(distance2))
  except AssertionError:
    raise AssertionError(f"Distances have different powers\n{distance1}\n{distance2}")
  try:
    np.testing.assert_array_equal(pscale(distance1), pscale(distance2))
  except AssertionError:
    raise AssertionError(f"Distances have different pscales\n{distance1}\n{distance2}")
  return np.testing.assert_allclose(pixels(distance1), pixels(distance2), *args, **kwargs)
