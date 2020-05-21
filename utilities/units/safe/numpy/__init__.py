import functools, numpy as np
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

def polyfit(x, y, deg):
  x = np.array(x)
  y = np.array(y)
  xpixels = pixels(x, power=None)
  ypixels = pixels(y, power=None)
  coeffs = np.polyfit(x=xpixels, y=ypixels, deg=deg)

  xpscale = {_pscale(_)[()] for _ in np.ravel(x) if _ and _pscale(_)[()] is not None}
  if len(xpscale) > 1: raise UnitsError(f"x has multiple different pscales {xpscale}")
  ypscale = {_pscale(_)[()] for _ in np.ravel(y) if _ and _pscale(_)[()] is not None}
  if len(ypscale) > 1: raise UnitsError(f"y has multiple different pscales {ypscale}")
  pscale = xpscale | ypscale
  if len(pscale) > 1: raise UnitsError(f"x and y have inconsistent pscales {pscale}")
  if not pscale: return coeffs
  pscale = pscale.pop()

  @functools.partial(np.vectorize, otypes=[object])
  def power(distance):
    if not distance: return None
    return _power(distance)[()]

  xpower = power(x)
  ypower = power(y)

  """
          A                B        C
                        --   --
  --               --   |[1]  |
  |[L^2] [L]   [1]  | @ |[L]  | = [L^2]
  --               --   |[L^2]|
                        --   --
  [1, x, x^2, ...] = A
  y = C
  will return B
  """

  if len(y.shape) > 1:
    raise NotImplementedError
  if len(set(xpower) - {None}) > 1 or len(set(ypower) - {None}) > 1:
    raise NotImplementedError

  ypower, = set(ypower) - {None}
  xpower, = set(xpower) - {None}

  constpower = ypower
  coeffpowers = [constpower - (len(coeffs)-i)*xpower for i in range(1, len(coeffs)+1)]

  return distances(pixels=coeffs, pscale=pscale, power=coeffpowers)

__all__ = [
  "fft", "linalg", "testing",
  "angle", "isclose", "linspace", "polyfit",
]
