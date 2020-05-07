import numpy as np
from ... import UnitsError
from ..core import _power, _pscale, distances, pixels
from . import fft, linalg, testing

@np.vectorize
def isclose(distance1, distance2, *args, **kwargs):
  try:
    np.testing.assert_array_equal(_power(distance1), _power(distance2))
  except AssertionError:
    raise UnitsError(f"Trying to compare distances with different powers\n{distance1}\n{distance2}")
  try:
    np.testing.assert_array_equal(_pscale(distance1), _pscale(distance2))
  except AssertionError:
    raise UnitsError(f"Trying to compare distances with different pscales\n{distance1}\n{distance2}")
  return np.isclose(pixels(distance1, pscale=_pscale(distance1), power=_power(distance1)), pixels(distance2, pscale=_pscale(distance2), power=_power(distance2)), *args, **kwargs)

def angle(distance, *args, **kwargs):
  return np.angle(pixels(distance), *args, **kwargs)

def linspace(start, stop, *args, **kwargs):
  stop - start
  pscale = _pscale(start)
  power = _power(start)

  startpixels = pixels(start, pscale=pscale, power=power)
  stoppixels = pixels(stop, pscale=pscale, power=power)
  resultpixels = np.linspace(startpixels, stoppixels, *args, **kwargs)
  return distances(pixels=resultpixels, pscale=pscale, power=power)
