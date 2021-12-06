import numba as nb, numpy as np, uncertainties as unc, uncertainties.umath as umath

def covariance_matrix(*args, **kwargs):
  """
  Covariance matrix for the uncertainties module
  that enforces symmetry (the normal one is symmetric up
  to rounding errors)
  """
  result = np.array(unc.covariance_matrix(*args, **kwargs))
  return (result + result.T) / 2

@nb.vectorize([nb.int64(nb.float64, nb.float64, nb.float64)])
def __floattoint(flt, atol, rtol):
  result = int(flt)
  flt = float(flt)
  for thing in result, result+1, result-1:
    #this version needs https://github.com/numba/numba/pull/6074 or https://github.com/numba/numba/pull/4610
    #if np.isclose(thing, flt, atol=atol, rtol=rtol):
    if np.abs(thing-flt) <= atol + rtol * np.abs(flt):
      return thing
  raise ValueError("not an int")

def floattoint(flt, *, atol=0, rtol=1e-10):
  """
  If flt is an integer within absolute and relative tolerance atol and rtol,
  return it as an int.  Otherwise raise an error.
  """
  try:
    return __floattoint(flt, atol, rtol)
  except ValueError as e:
    if str(e) == "not an int":
      raise ValueError(f"not an int: {flt}")
    raise

def weightedaverage(a, *args, **kwargs):
  """
  Weighted average of an array a, where the weights are 1/error^2.
  a should contain objects from the uncertainties module.
  """
  from . import units
  return np.average(units.nominal_values(a), weights=1/units.std_devs(a)**2, *args, **kwargs)

def weightedvariance(a, *, subtractaverage=True):
  """
  Weighted variance of an array a, where the weights are 1/error^2.
  a should contain objects from the uncertainties module.
  """
  from . import units
  if subtractaverage:
    average = weightedaverage(a)
    if not isinstance(a, np.ndarray): a = np.array(a)
    a -= average
  return np.average(units.nominal_values(a)**2, weights=1/units.std_devs(a)**2)

def weightedstd(*args, **kwargs):
  """
  Weighted standard deviation of an array a, where the weights are 1/error^2.
  a should contain objects from the uncertainties module.
  """
  return weightedvariance(*args, **kwargs) ** 0.5

def sorted_eig(*args, **kwargs):
  val, vec = np.linalg.eig(*args, **kwargs)
  order = np.argsort(val)[::-1]
  return val[order], vec[:,order]

def affinetransformation(scale=None, rotation=None, shear=None, translation=None):
  """
  https://github.com/scikit-image/scikit-image/blob/f0fcdc8d73c20d741e31e5d57efad1a38426f7fd/skimage/transform/_geometric.py#L876-L880
  """
  if scale is None: scale = 1
  if np.isscalar(scale): scale = scale, scale
  sx, sy = scale

  if rotation is None: rotation = 0
  if shear is None: shear = 0

  if translation is None: translation = 0
  if np.isscalar(translation): translation = translation, translation
  tx, ty = translation

  return np.array([
    [sx * umath.cos(rotation), -sy * umath.sin(rotation + shear), tx],
    [sx * umath.sin(rotation),  sy * umath.cos(rotation + shear), ty],
    [                       0,                                 0, 1],
  ])
