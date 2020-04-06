import functools, itertools, numpy as np, uncertainties as unc, uncertainties.unumpy as unp

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

def correlateddistances(*, pscale=None, pixels=None, microns=None, distances=None, covariance=None, power=None):
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

  if power is None and distances is None:
    power = 1
  if distances is not None:
    distpower = {_.power for _ in distances}
    if covariance is not None: distpower |= {_.power/2 for _ in np.ravel(covariance)}
    if len(distpower) > 1: raise UnitsError(f"Provided distances with multiple powers {distpower}")
    distpower = distpower.pop()
    if distpower != power is not None: raise UnitsError("Provided both power and distances, but they're inconsistent")
    power = distpower

  if distances is not None:
    pixels = __pixels(distances)
    if covariance is not None:
      covariance = __pixels(covariance)
  if pixels is not None: name = "pixels"; values = pixels
  if microns is not None: name = "microns"; values = microns

  if covariance is not None:
    values = unc.correlated_values(values, covariance)

  return tuple(Distance(pscale=pscale, power=power, **{name: _}) for _ in values)

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
def nominal_values(distance):
  return distance.nominal_value
@np.vectorize
def std_devs(distance):
  return distance.std_dev

def inv(matrix):
  invpixels = np.linalg.inv(pixels(matrix))

  pscale = {_.pscale for _ in np.ravel(matrix) if _}
  if len(pscale) > 1: raise UnitsError(f"matrix has multiple different pscales {pscale}")
  pscale = pscale.pop()

  @functools.partial(np.vectorize, otypes=[object])
  def minuspower(distance):
    if not distance: return None
    return -distance.power
  power = minuspower(matrix.T)

  while any(_ is None for _ in np.ravel(power)):
    progressed = False
    for (i, j), element in np.ndenumerate(power):
      if element is not None: continue
      for (ii, jj), element2 in np.ndenumerate(power):
        if element2 is not None and power[i,jj] is not None and power[ii,j] is not None:
          power[i,j] = power[ii,j] + power[i,jj] - element2
          progressed = True
          break
    if not progressed:
      raise UnitsError(f"Can't figure out any more powers in power matrix {power}")

  return distances_differentpowers(pixels=invpixels, pscale=pscale, power=power, defaulttozero=True)

def solve(matrix, vector):
  try:
    length, = vector.shape
    assert matrix.shape == (length, length)
  except Exception:
    raise NotImplementedError("This version of solve only works for an NxN matrix and an N length vector")

  solvepixels = np.linalg.solve(pixels(matrix), pixels(vector))

  matrixpscale = {_.pscale for _ in np.ravel(matrix) if _}
  if len(matrixpscale) > 1: raise UnitsError(f"matrix has multiple different pscales {matrixpscale}")
  vectorpscale = {_.pscale for _ in np.ravel(vector) if _}
  if len(vectorpscale) > 1: raise UnitsError(f"vector has multiple different pscales {vectorpscale}")
  pscale = matrixpscale | vectorpscale
  if len(pscale) > 1: raise UnitsError(f"matrix and vector have inconsistent pscales {pscale}")
  pscale = pscale.pop()

  @functools.partial(np.vectorize, otypes=[object])
  def power(distance):
    if not distance: return None
    return distance.power

  matrixpower = power(matrix)
  vectorpower = power(vector)

  """
          A                B        C
  --               --   -- --   --   --
  |[L]   [L]   [1]  |   |[1]|   |[L]  |
  |[L^2] [L^2] [L]  | @ |[1]| = |[L^2]|
  |[L^3] [L^3] [L^2]|   |[L]|   |[L^3]|
  --               --   -- --   --   --
  matrix = A
  vector = C
  will return B
  Have to make sure that each column of A looks like C
  """

  resultpowers = []
  for i, columnpower in enumerate(matrixpower.T):
    resultentrypower = {v-m for m, v in zip(columnpower, vectorpower) if m is not None and v is not None}
    if len(resultentrypower) > 1:
      raise UnitsError(f"column {i} of matrix has powers {columnpower} which aren't consistent with the vector powers {vectorpower}")
    resultpowers.append(resultentrypower.pop())

  return distances_differentpowers(pixels=solvepixels, pscale=pscale, power=resultpowers)
