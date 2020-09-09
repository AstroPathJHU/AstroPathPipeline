import dataclasses, itertools, numpy as np

from ..baseclasses.overlap import OverlapCollection
from ..utilities import units
from ..utilities.misc import dummylogger
from ..utilities.tableio import writetable
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

def stitchlayers(*args, usecvxpy=False, **kwargs):
  return (__stitchlayerscvxpy if usecvxpy else __stitchlayers)(*args, **kwargs)

def __stitchlayers(*, overlaps, logger=dummylogger):
  layers = sorted(set.union(*(set(o.layers) for o in overlaps)))
  size = 2*len(layers)  #2* for x and y
  A = np.zeros(shape=(size, size), dtype=units.unitdtype)
  b = np.zeros(shape=(size,), dtype=units.unitdtype)
  c = 0
  ld = {layer: i for i, layer in enumerate(layers)}

  alloverlaps = overlaps
  overlaps = [o for o in overlaps if not o.result.exit]
  for o in overlaps[:]:
    if o.layer2 > o.layer1 and any(o.p1 == oo.p1 and o.p2 == oo.p2 and (oo.layer2, oo.layer1) == (o.layer1, o.layer2) for oo in overlaps):
      overlaps.remove(o)

  for o in overlaps:
    ix = 2*ld[o.layer1]
    iy = 2*ld[o.layer1]+1
    jx = 2*ld[o.layer2]
    jy = 2*ld[o.layer2]+1
    assert ix >= 0, ix
    assert iy < 2*len(layers), iy
    assert jx >= 0, jx
    assert jy < 2*len(layers), jy

    ii = np.ix_((ix,iy), (ix,iy))
    ij = np.ix_((ix,iy), (jx,jy))
    ji = np.ix_((jx,jy), (ix,iy))
    jj = np.ix_((jx,jy), (jx,jy))
    inversecovariance = units.np.linalg.inv(o.result.covariance)

    A[ii] += inversecovariance
    A[ij] -= inversecovariance
    A[ji] -= inversecovariance
    A[jj] += inversecovariance

    i = np.ix_((ix, iy))
    j = np.ix_((jx, jy))

    constpiece = (-units.nominal_values(o.result.dxvec))

    b[i] += 2 * inversecovariance @ constpiece
    b[j] -= 2 * inversecovariance @ constpiece

    c += constpiece @ inversecovariance @ constpiece

  logger.debug("assembled A b c")

  result = units.np.linalg.solve(2*A, -b)

  logger.debug("solved quadratic equation")

  delta2nllfor1sigma = 1

  covariancematrix = units.np.linalg.inv(A) * delta2nllfor1sigma
  logger.debug("got covariance matrix")
  result = np.array(units.correlated_distances(distances=result, covariance=covariancematrix))

  x = result.reshape(len(layers), 2)

  logger.debug("done")

  return LayerStitchResult(overlaps=alloverlaps, x=x, A=A, b=b, c=c, logger=logger)

def __stitchlayerscvxpy(*, overlaps, logger=dummylogger):
  try:
    import cvxpy as cp
  except ImportError:
    raise ImportError("To stitch with cvxpy, you have to install cvxpy")

  layers = sorted(set.union(*(set(o.layers) for o in overlaps)))

  pscale, = {_.pscale for _ in overlaps}

  x = cp.Variable(shape=len(layers), 2)

  twonll = 0
  alloverlaps = overlaps
  overlaps = [o for o in overlaps if not o.result.exit]
  for o in overlaps[:]:
    if o.layer2 > o.layer1 and any(o.p1 == oo.p1 and o.p2 == oo.p2 and (oo.layer2, oo.layer1) == (o.layer1, o.layer2) for oo in overlaps):
      overlaps.remove(o)

  layerx = {layer: xx for layer, xx in itertools.zip_longest(layers, x)}

  for o in overlaps:
    x1 = layerx[o.layer1]
    x2 = layerx[o.layer2]
    twonll += cp.quad_form(x1 - x2 + units.pixels(-units.nominal_values(o.result.dxvec), pscale=pscale, power=1))

  minimize = cp.Minimize(twonll)
  prob = cp.Problem(minimize)
  prob.solve()

  return LayerStitchResultCvxpy(overlaps=alloverlaps, x=x, problem=prob, pscale=pscale, logger=logger)

class LayerStitchResultBase(OverlapCollection):
  def __init__(self, *, overlaps, x, logger=dummylogger):
    self.__overlaps = overlaps
    self.__logger = logger
    self.__x = x

  @property
  def pscale(self):
    pscale, = {_.pscale for _ in self.overlaps}
    return pscale

  @property
  def overlaps(self): return self.__overlaps

  @property
  def x(self): return self.__x

  @property
  def layerpositions(self):
    result = []
    for layer, row in itertools.zip_longest(self.layers, self.x):
      x, y = units.nominal_values(row)
      (cov_x_x, cov_x_y), (cov_x_y, cov_y_y) = units.covariance_matrix(*row)
      result.append(
        LayerPosition(
          n=layer,
          x=x,
          y=y,
          cov_x_x=cov_x_x,
          cov_x_y=cov_x_y,
          cov_y_y=cov_y_y,
        )
      )
    return result

  @property
  def layerpositioncovariances(self):
    result = []
    for (layer1, row1), (layer2, row2) in itertools.product(itertools.zip_longest(self.layers, self.x), repeat=2):
      if layer1 == layer2: continue
      cov = np.array(units.covariance_matrix(*row1, *row2))
      result.append(
        LayerPositionCovariance(
          n1=layer1,
          n2=layer2,
          cov_x1_x2=cov[0,2],
          cov_x1_y2=cov[0,3],
          cov_y1_x2=cov[1,2],
          cov_y1_y2=cov[1,3],
        )
      )
    return result

  def writetable(self, *filenames):
    positionfilename, covariancefilename = filenames
    writetable(positionfilename, self.layerpositions)
    writetable(covariancefilename, self.layerpositioncovariances)

class LayerStitchResult(LayerStitchResultBase):
  def __init__(self, *args, A, b, c, **kwargs):
    self.A = A
    self.b = b
    self.c = c
    super().__init__(*args, **kwargs)

class LayerStitchResultCvxpy(LayerStitchResultBase):
  def __init__(self, *, x, problem, pscale, **kwargs):
    super().__init__(
      x=units.distances(pixels=x.value, pscale=pscale),
      **kwargs
    )
    self.problem = problem
    self.xvar = x
    self.Tvar = T


@dataclasses.dataclass(frozen=True)
class LayerPosition(DataClassWithDistances):
  pixelsormicrons = "pixels"

  n: int
  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  cov_x_x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  cov_x_y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  cov_y_y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False

@dataclasses.dataclass(frozen=True)
class LayerPositionCovariance(DataClassWithDistances):
  pixelsormicrons = "pixels"

  n1: int
  n2: int
  cov_x1_x2: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  cov_x1_y2: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  cov_y1_x2: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  cov_y1_y2: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False
