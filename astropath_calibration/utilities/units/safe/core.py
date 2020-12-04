from ..core import UnitsError

import collections, itertools, numbers, numpy as np, uncertainties as unc

class Distance:
  def __new__(cls, *args, pixels=None, microns=None, centimeters=None, defaulttozero=False, **kwargs):
    if defaulttozero and {pixels, microns, centimeters} == {None, 0}:
      return 0.
    return super().__new__(cls)

  def __init__(self, *, pscale, pixels=None, microns=None, centimeters=None, power=1, defaulttozero=False):
    if (pixels is not None) + (microns is not None) + (centimeters is not None) != 1:
      raise TypeError("Have to provide exactly one of pixels, microns, or centimeters")
    if centimeters is not None:
      microns = centimeters * (1e4**power if power and centimeters else 1)
    if microns is not None:
      pixels = microns * (pscale**power if power and microns else 1)
    if isinstance(pixels, Distance):
      power += pixels._power
      pixels = pixels._pixels
    if isinstance(pscale, Distance): pscale = pscale.asdimensionless
    if isinstance(power, Distance): power = power.asdimensionless

    if power is None and (pixels or microns or centimeters):
      raise ValueError("Can't set power=None")
    if not power or pixels == 0 or microns == 0: pscale = None

    self.__pscale = pscale
    self.__power = power
    self.__pixels = pixels
    self.__microns = pixels / (pscale**power if power and pixels else 1)

  @property
  def _microns(self):
    return self.__microns
  @property
  def _pixels(self):
    return self.__pixels
  @property
  def _pscale(self):
    return self.__pscale
  @property
  def _power(self):
    return self.__power

  def __add__(self, other):
    if isinstance(other, np.ndarray):
      return other + self
    if not other: return self
    if not self: return other
    if self._power == 0: return self.asdimensionless + other
    if not hasattr(other, "_pscale"): return NotImplemented
    if self._power != other._power: raise UnitsError(f"Trying to add distances with different powers {self._power} {other._power}")
    if None is not self._pscale != other._pscale is not None: raise UnitsError(f"Trying to add distances with different pscales {self._pscale} {other._pscale}")
    pscale = self._pscale if self._pscale is not None else other._pscale
    return Distance(pscale=pscale, power=self._power, pixels=self._pixels+other._pixels)
  def __radd__(self, other):
    return self + other
  def __mul__(self, other):
    if isinstance(other, Distance):
      if not self or not other: return 0.
      if None is not self._pscale != other._pscale is not None: raise UnitsError("Trying to multiply distances with different pscales")
      pscale = self._pscale if self._pscale is not None else other._pscale
      return Distance(pscale=pscale, power=self._power+other._power, pixels=self._pixels*other._pixels)
    if isinstance(other, np.ndarray):
      return other * self
    return Distance(pscale=self._pscale, power=self._power, pixels=other*self._pixels)
  def __rmul__(self, other): return self * other
  def __sub__(self, other): return self + -other
  def __rsub__(self, other): return other + -self
  def __pos__(self): return 1*self
  def __neg__(self): return -1*self
  def __truediv__(self, other): return self * (1/other)
  def __rtruediv__(self, other):
    oneoverself = Distance(pscale=self._pscale, power=-self._power, pixels=1/self._pixels)
    return other * oneoverself
  def __floordiv__(self, other):
    return float(self / other) // 1
  def __mod__(self, other):
    return self - self // other * other
  def __pow__(self, other):
    return Distance(pscale=self._pscale, power=self._power*other, pixels=self._pixels**other)
  def __bool__(self): return bool(self._pixels)
  @property
  def asdimensionless(self):
    if self and self._power != 0: raise ValueError("Can only convert Distance to dimensionless if pixels == microns == 0 or power == 0")
    assert self._pixels == self._microns
    return self._pixels
  def __float__(self):
    return float(self.asdimensionless)
  def __int__(self):
    return int(float(self))
  def __eq__(self, other):
    return pixels(self - other, power=None) == 0
  def __lt__(self, other):
    return pixels(self - other, power=None) < 0
  def __le__(self, other):
    return pixels(self - other, power=None) <= 0
  def __gt__(self, other):
    return pixels(self - other, power=None) > 0
  def __ge__(self, other):
    return pixels(self - other, power=None) >= 0
  def __abs__(self):
    return Distance(pscale=self._pscale, power=self._power, pixels=abs(self._pixels))
  def __hash__(self):
    if self._power == 0: return hash(float(self))
    return hash((float(self._power), float(self._pscale), float(self._pixels)))

  def sqrt(self): return self**0.5
  def conjugate(self):
    try:
      return Distance(pscale=self._pscale, power=self._power, pixels=np.conjugate(self._pixels))
    except (TypeError, AttributeError):
      if np.conjugate(self.nominal_value) != self.nominal_value: raise AssertionError
      return self

  @property
  def nominal_value(self): return Distance(pscale=self._pscale, power=self._power, pixels=unc.nominal_value(self._pixels))
  n = nominal_value
  @property
  def std_dev(self): return Distance(pscale=self._pscale, power=self._power, pixels=unc.std_dev(self._pixels))
  s = std_dev
  @property
  def derivatives(self): return {k: Distance(pscale=self._pscale, power=self._power, pixels=v) for k, v in self._pixels.derivatives.items()}

  def __repr__(self):
    if self._pscale is None: assert self._pixels == self._microns; return repr(self._pixels)
    return f"Distance(pscale={self._pscale}, pixels={self._pixels}, power={self._power})"
  def __str__(self):
    if self._power == 0: return str(self._pixels)
    if self._power == 1: return f"{self._pixels} pixels"
    return f"{self._pixels} pixels^{self._power}"

distances = np.vectorize(Distance, excluded=["pscale"], otypes=[object])
__distances = distances #for use in functions with a distances kwarg
  
def correlated_distances(*, pscale=None, pixels=None, microns=None, distances=None, covariance=None, power=None):
  if (pixels is not None) + (microns is not None) + (distances is not None) != 1:
    raise TypeError("Have to provide exactly one of pixels, microns, or distances")

  if pscale is None and distances is None and not np.all(power == 0):
    raise TypeError("If you don't provide distances, you have to provide pscale")
  if distances is not None:
    distpscale = {_._pscale for _ in itertools.chain(distances, np.ravel(covariance) if covariance is not None else []) if _ and _pscale(_)[()] is not None}
    if not distpscale: distpscale = {None}
    if len(distpscale) > 1: raise UnitsError("Provided distances with multiple pscales")
    distpscale = distpscale.pop()
    if None is not distpscale != pscale is not None: raise UnitsError("Provided both pscale and distances, but they're inconsistent")
    if distpscale is not None: pscale = distpscale

  if distances is not None:
    pixels = __pixels(distances, power=None)
    if covariance is not None:
      covariance = np.array(covariance)
      distcovariance = covariance
      covariance = __pixels(covariance, power=None)
  if pixels is not None: name = "pixels"; values = pixels
  if microns is not None: name = "microns"; values = microns

  values = np.array(values)
  try:
    length, = values.shape
    if covariance is not None:
      assert covariance.shape == (length, length)
  except Exception:
    raise TypeError("Need to give values of length N and an NxN covariance matrix")

  if covariance is not None:
    values = unc.correlated_values(values, covariance)

  if power is None and distances is None:
    power = 1
  if power is not None and not isinstance(power, collections.abc.Sequence):
    power = np.array([power] * length)

  if distances is not None:
    distpower = np.array([_power(_)[()] if _ else None for _ in distances])
    covpower = _power(distcovariance)
    if covariance is not None:
      repeat = True
      while repeat:
        repeat = False
        for (i1, p1), (i2, p2) in itertools.product(enumerate(distpower), repeat=2):
          if not distcovariance[i1,i2]: continue
          if p1 is None and p2 is None and i1 != i2: continue
          if p1 is None and covpower[i1,i2] is None: continue
          if p2 is None and covpower[i1,i2] is None: continue
          if i1 == i2 and p1 is None: distpower[i1] = covpower[i1,i2] / 2; repeat = True; break
          if p1 is None: distpower[i1] = covpower[i1,i2] - p2; repeat = True; break
          if p2 is None: distpower[i2] = covpower[i1,i2] - p1; repeat = True; break
          if covpower[i1,i2] != p1+p2:
            raise UnitsError(f"Covariance entry {i1},{i2} has power {covpower[i1,i2]}, should be {p1}+{p2}")
    if power is not None and not np.all(power == distpower):
      raise UnitsError(f"Provided both power and distances, but they're inconsistent:\n{power}\n{distpower}")
    power = distpower

  if len(power) != length:
    raise TypeError(f"power has the wrong length {len(power)}, should be {length}")

  return tuple(Distance(pscale=pscale, power=p, **{name: v}) for v, p in zip(values, power))

@np.vectorize
def pixels(distance, *, pscale=None, power=1):
  if not distance: return 0.
  if isinstance(pscale, Distance): pscale = pscale.asdimensionless
  if None is not pscale != _pscale(distance)[()] is not None:
    raise ValueError(f"Inconsistent pscales {pscale} {_pscale(distance)}")
  if None is not power != _power(distance)[()] is not None:
    raise ValueError(f"Inconsistent powers {power} {_power(distance)}")
  if isinstance(distance, (numbers.Number, unc.core.AffineScalarFunc)): return distance
  return distance._pixels

__pixels = pixels #for use in functions with a pixels kwarg

@np.vectorize
def microns(distance, *, pscale=None, power=1):
  if not distance: return 0.
  if None is not pscale != _pscale(distance)[()] is not None:
    raise ValueError(f"Inconsistent pscales {pscale} {_pscale(distance)}")
  if None is not power != _power(distance)[()] is not None:
    raise ValueError(f"Inconsistent powers {power} {_power(distance)}")
  if isinstance(distance, (numbers.Number, unc.core.AffineScalarFunc)): return distance
  return distance._microns

@np.vectorize
def asdimensionless(distance):
  if isinstance(distance, (numbers.Number, unc.core.AffineScalarFunc)): return distance
  return distance.asdimensionless

@np.vectorize
def _power(distance):
  if isinstance(distance, (numbers.Number, unc.core.AffineScalarFunc)): return 0.
  return distance._power
@np.vectorize
def _pscale(distance):
  if isinstance(distance, (numbers.Number, unc.core.AffineScalarFunc)) or not distance: return None
  return distance._pscale

def nominal_value(distance):
  if isinstance(distance, numbers.Number): return distance
  return distance.nominal_value
nominal_values = np.vectorize(nominal_value)
def std_dev(distance):
  if isinstance(distance, numbers.Number): return 0.
  return distance.std_dev
std_devs = np.vectorize(std_dev)

def covariance_matrix(distances):
  pixels = __pixels(distances, power=None)
  covpixels = unc.covariance_matrix(pixels)

  pscale = {_._pscale for _ in distances if _._pscale is not None}
  if not pscale: pscale = {None}
  if len(pscale) > 1: raise UnitsError("Provided distances with multiple pscales")
  pscale = pscale.pop()

  distpowers = [_._power for _ in distances]
  covpowers = [[a + b for b in distpowers] for a in distpowers]

  return __distances(pixels=covpixels, pscale=pscale, power=covpowers)
