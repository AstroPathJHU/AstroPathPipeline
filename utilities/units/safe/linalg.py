import functools, itertools, numpy as np
from .core import distances, pixels, UnitsError

def inv(matrix):
  invpixels = np.linalg.inv(pixels(matrix))

  pscale = {_._pscale for _ in np.ravel(matrix) if _ and _._pscale is not None}
  if not pscale: return invpixels
  if len(pscale) > 1: raise UnitsError(f"matrix has multiple different pscales {pscale}")
  pscale = pscale.pop()

  @functools.partial(np.vectorize, otypes=[object])
  def minuspower(distance):
    if not distance: return None
    return -distance._power
  power = minuspower(matrix.T)

  for (i, row), (j, row2) in itertools.combinations(enumerate(power.T), 2):
    if len({p2-p1 for p1, p2 in zip(row, row2) if p1 is not None is not p2}) > 1:
      raise ValueError(f"Rows {i} and {j} of the matrix aren't compatible in powers:\n{matrix[i]}\n{matrix[j]}")

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

  return distances(pixels=invpixels, pscale=pscale, power=power, defaulttozero=True)

def solve(matrix, vector):
  try:
    length, = vector.shape
    assert matrix.shape == (length, length)
  except Exception:
    raise NotImplementedError("This version of solve only works for an NxN matrix and an N length vector")

  solvepixels = np.linalg.solve(pixels(matrix), pixels(vector))

  matrixpscale = {_._pscale for _ in np.ravel(matrix) if _ and _._pscale is not None}
  if len(matrixpscale) > 1: raise UnitsError(f"matrix has multiple different pscales {matrixpscale}")
  vectorpscale = {_._pscale for _ in np.ravel(vector) if _ and _._pscale is not None}
  if len(vectorpscale) > 1: raise UnitsError(f"vector has multiple different pscales {vectorpscale}")
  pscale = matrixpscale | vectorpscale
  if len(pscale) > 1: raise UnitsError(f"matrix and vector have inconsistent pscales {pscale}")
  if not pscale: return solvepixels
  pscale = pscale.pop()

  @functools.partial(np.vectorize, otypes=[object])
  def power(distance):
    if not distance: return None
    return distance._power

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

  return distances(pixels=solvepixels, pscale=pscale, power=resultpowers)
