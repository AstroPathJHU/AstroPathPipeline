import numpy as np
from .core import Distance, pixels, _power, _pscale

def assert_allclose(distance1, distance2, *args, **kwargs):
  try:
    np.testing.assert_array_equal(_power(distance1), _power(distance2))
  except AssertionError:
    raise AssertionError(f"Distances have different powers\n{distance1}\n{distance2}")
  try:
    np.testing.assert_array_equal(_pscale(distance1), _pscale(distance2))
  except AssertionError:
    raise AssertionError(f"Distances have different pscales\n{distance1}\n{distance2}")
  return np.testing.assert_allclose(pixels(distance1), pixels(distance2), *args, **kwargs)
