import collections, functools, itertools, numpy as np, uncertainties as unc, uncertainties.unumpy as unp

class UnitsError(Exception): pass

class Distance:
  def __new__(cls, *args, pixels=None, microns=None, defaulttozero=False, **kwargs):
    if defaulttozero and (pixels == 0 and microns is None or microns == 0 and pixels is None):
      return 0
    return super().__new__(cls)

  def __init__(self, *, pscale, pixels=None, microns=None, power=1, defaulttozero=False):
    self.__pscale = pscale
    self.__power = power
    if (pixels is not None) == (microns is not None):
      raise TypeError("Have to provide exactly one of pixels or microns")
    if pixels is not None:
      self.__pixels = pixels
      self.__microns = pixels / pscale**power
    if microns is not None:
      self.__microns = microns
      self.__pixels = microns * pscale**power

  @property
  def microns(self):
    return self.__microns
  @property
  def pixels(self):
    return self.__pixels
  @property
  def pscale(self):
    return self.__pscale
  @property
  def power(self):
    return self.__power

  def __add__(self, other):
    if other == 0: return self
    if not hasattr(other, "pscale"): return NotImplemented
    if self.pscale != other.pscale: raise UnitsError("Trying to add distances with different pscales")
    if self.power != other.power: raise UnitsError("Trying to add distances with different powers")
    return Distance(pscale=self.pscale, power=self.power, pixels=self.pixels+other.pixels)
  def __radd__(self, other):
    return self + other
  def __mul__(self, other):
    if isinstance(other, Distance):
      if self.pscale != other.pscale: raise UnitsError("Trying to multiply distances with different pscales")
      return Distance(pscale=self.pscale, power=self.power+other.power, pixels=self.pixels*other.pixels)
    return Distance(pscale=self.pscale, power=self.power, pixels=other*self.pixels)
  def __rmul__(self, other): return self * other
  def __sub__(self, other): return self + -other
  def __rsub__(self, other): return other + -self
  def __pos__(self): return 1*self
  def __neg__(self): return -1*self
  def __truediv__(self, other): return self * (1/other)
  def __rtruediv__(self, other):
    oneoverself = Distance(pscale=self.pscale, power=-self.power, pixels=1/self.pixels)
    return other * oneoverself
  def __pow__(self, other):
    return Distance(pscale=self.pscale, power=self.power*other, pixels=self.pixels**other)
  def __float__(self):
    if self.power != 0: raise ValueError("Can only convert float to distance if power == 0")
    assert self.pixels == self.microns
    return float(self.pixels)
  def sqrt(self): return self**0.5

  @property
  def nominal_value(self): return Distance(pscale=self.pscale, power=self.power, pixels=self.pixels.nominal_value)
  @property
  def n(self): return Distance(pscale=self.pscale, power=self.power, pixels=self.pixels.n)
  @property
  def std_dev(self): return Distance(pscale=self.pscale, power=self.power, pixels=self.pixels.std_dev)
  @property
  def s(self): return Distance(pscale=self.pscale, power=self.power, pixels=self.pixels.s)
  @property
  def derivatives(self): return {k: Distance(pscale=self.pscale, power=self.power, pixels=v) for k, v in self.pixels.derivatives.items()}

distances = np.vectorize(Distance, excluded=["pscale", "power"])
distances_differentpowers = np.vectorize(Distance, excluded=["pscale"])
  
def udistance(*, pscale, pixels=None, microns=None, power=1):
  """
  return unc.ufloat(
    Distance(
      pixels=pixels.n if pixels is not None else None,
      microns=microns.n if microns is not None else None,
      pscale=pscale, power=power,
    ),
    Distance(
      pixels=pixels.s if pixels is not None else None,
      microns=microns.s if microns is not None else None,
      pscale=pscale, power=power,
    ),
  )
  """
  return Distance(pscale=pscale, pixels=pixels, microns=microns, power=power)

def correlated_distances(*, pscale=None, pixels=None, microns=None, distances=None, covariance=None, power=None):
  if (pixels is not None) + (microns is not None) + (distances is not None) != 1:
    raise TypeError("Have to provide exactly one of pixels, microns, or distances")

  if pscale is None and distances is None:
    raise TypeError("If you don't provide distances, you have to provide pscale")
  if distances is not None:
    distpscale = {_.pscale for _ in itertools.chain(distances, np.ravel(covariance) if covariance is not None else [])}
    if len(distpscale) > 1: raise UnitsError("Provided distances with multiple pscales")
    distpscale = distpscale.pop()
    if distpscale != pscale is not None: raise UnitsError("Provided both pscale and distances, but they're inconsistent")
    pscale = distpscale

  if distances is not None:
    pixels = __pixels(distances)
    if covariance is not None:
      covariance = np.array(covariance)
      distcovariance = covariance
      covariance = __pixels(covariance)
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
    distpower = np.array([_.power for _ in distances])
    if covariance is not None:
      for (i1, p1), (i2, p2) in itertools.product(enumerate(distpower), repeat=2):
        if distcovariance[i1,i2].power != p1+p2:
          raise UnitsError(f"Covariance entry {i1},{i2} has power {covariance[i1,i2].power}, should be {p1}+{p2}")
    if power is not None and not np.all(power == distpower):
      raise UnitsError(f"Provided both power and distances, but they're inconsistent:\n{power}\n{distpower}")
    power = distpower

  if len(power) != length:
    raise TypeError(f"power has the wrong length {len(power)}, should be {length}")

  return tuple(Distance(pscale=pscale, power=p, **{name: v}) for v, p in zip(values, power))

@np.vectorize
def pixels(distance):
  if distance == 0: return distance
  return distance.pixels
__pixels = pixels #for use in functions with a pixels kwarg
@np.vectorize
def microns(distance):
  if distance == 0: return distance
  return distance.microns

@np.vectorize
def nominal_value(distance):
  return distance.nominal_value
nominal_values = nominal_value
@np.vectorize
def std_dev(distance):
  return distance.std_dev
std_devs = std_dev

def covariance_matrix(distances):
  pixels = __pixels(distances)
  covpixels = unc.covariance_matrix(pixels)

  pscale = {_.pscale for _ in distances}
  if len(pscale) > 1: raise UnitsError("Provided distances with multiple pscales")
  pscale = pscale.pop()

  distpowers = [_.power for _ in distances]
  covpowers = [[a.power + b.power for b in distances] for a in distances]

  return distances_differentpowers(pixels=covpixels, pscale=pscale, power=covpowers)
