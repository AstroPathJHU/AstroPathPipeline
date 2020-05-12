import numpy as np
from ..core import pixels, _power, _pscale

def assert_allclose(distance1, distance2, *args, **kwargs):
  distance1 = np.array(distance1)
  distance2 = np.array(distance2)
  try:
    if np.any(distance1) or np.any(distance2):
      np.testing.assert_array_equal(_power(distance1[distance1!=0]), _power(distance2[distance2!=0]))
  except AssertionError:
    raise AssertionError(f"Distances have different powers\n{distance1}\n{distance2}")
  try:
    np.testing.assert_array_equal(_pscale(distance1), _pscale(distance2))
  except AssertionError:
    raise AssertionError(f"Distances have different pscales\n{distance1}\n{distance2}")
  return np.testing.assert_allclose(pixels(distance1, pscale=_pscale(distance1), power=_power(distance1)), pixels(distance2, pscale=_pscale(distance2), power=_power(distance2)), *args, **kwargs)
