import dataclasses, itertools, numpy as np

from ..baseclasses.overlap import OverlapCollection
from ..utilities import units
from ..utilities.misc import dummylogger
from ..utilities.tableio import writetable
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

def stitchlayers(*args, usecvxpy=False, **kwargs):
  return (__stitchlayerscvxpy if usecvxpy else __stitchlayers)(*args, **kwargs)

def __stitchlayers(*, overlaps, eliminatelayer=0, filteroverlaps=lambda o: True, logger=dummylogger):
  layers = sorted(set.union(*(set(o.layers) for o in overlaps)))
  size = 2*len(layers)-2  #2* for x and y, - 2 for global shift
  A = np.zeros(shape=(size, size), dtype=units.unitdtype)
  b = np.zeros(shape=(size,), dtype=units.unitdtype)
  c = 0
  ld = {layer: i for i, layer in enumerate(layer for layer in layers if layer != layers[eliminatelayer])}
  ld[layers[eliminatelayer]] = None

  alloverlaps = overlaps
  overlaps = [o for o in overlaps if filteroverlaps(o) and not o.result.exit]
  for o in overlaps[:]:
    hasinverse = any(o.p1 == oo.p1 and o.p2 == oo.p2 and (oo.layer2, oo.layer1) == (o.layer1, o.layer2) for oo in overlaps)
    if ld[o.layer1] is None:
      assert hasinverse
      overlaps.remove(o)
    elif ld[o.layer2] is not None and o.layer2 > o.layer1:
      if hasinverse:
        overlaps.remove(o)

  for o in overlaps:
    layer1, layer2 = o.layers

    inversecovariance = units.np.linalg.inv(o.result.covariance)
    constpiece = (-units.nominal_values(o.result.dxvec))

    ix = 2*ld[layer1]
    iy = 2*ld[layer1]+1
    assert ix >= 0, ix
    assert iy < 2*len(layers), iy

    ii = np.ix_((ix,iy), (ix,iy))
    A[ii] += inversecovariance

    i = np.ix_((ix, iy))
    b[i] += 2 * inversecovariance @ constpiece

    c += constpiece @ inversecovariance @ constpiece

    if ld[layer2] is not None:
      jx = 2*ld[layer2]
      jy = 2*ld[layer2]+1
      assert jx >= 0, jx
      assert jy < 2*len(layers), jy

      ij = np.ix_((ix,iy), (jx,jy))
      ji = np.ix_((jx,jy), (ix,iy))
      jj = np.ix_((jx,jy), (jx,jy))

      A[ij] -= inversecovariance
      A[ji] -= inversecovariance
      A[jj] += inversecovariance

      j = np.ix_((jx, jy))
      b[j] -= 2 * inversecovariance @ constpiece

    else:
      for layer in ld:
        if layer == layer2: continue
        jx = 2*ld[layer]
        jy = 2*ld[layer]+1
        assert jx >= 0, jx
        assert jy < 2*len(layers), jy

        ij = np.ix_((ix,iy), (jx,jy))
        ji = np.ix_((jx,jy), (ix,iy))

        A[ij] += inversecovariance
        A[ji] += inversecovariance

        for otherlayer in ld:
          if otherlayer == layer2: continue
          kx = 2*ld[otherlayer]
          ky = 2*ld[otherlayer]+1
          assert kx >= 0, kx
          assert ky < 2*len(layers), ky
          jk = np.ix_((jx,jy), (kx,ky))
          A[jk] += inversecovariance

        j = np.ix_((jx, jy))
        b[j] += 2 * inversecovariance @ constpiece

  logger.debug("assembled A b c")

  result = units.np.linalg.solve(2*A, -b)

  logger.debug("solved quadratic equation")

  delta2nllfor1sigma = 1

  covariancematrix = units.np.linalg.inv(A) * delta2nllfor1sigma
  logger.debug("got covariance matrix")
  result = np.array(units.correlated_distances(distances=result, covariance=covariancematrix))

  x = result.reshape(len(layers)-1, 2)
  x = np.insert(x, eliminatelayer, [-np.sum(x, axis=0)], axis=0)

  logger.debug("done")

  return LayerStitchResult(overlaps=alloverlaps, x=x, A=A, b=b, c=c, logger=logger)

def __stitchlayerscvxpy(*, overlaps, logger=dummylogger):
  try:
    import cvxpy as cp
  except ImportError:
    raise ImportError("To stitch with cvxpy, you have to install cvxpy")

  layers = sorted(set.union(*(set(o.layers) for o in overlaps)))

  pscale, = {_.pscale for _ in overlaps}

  x = cp.Variable(shape=(len(layers), 2))

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
    twonll += cp.quad_form(
      x1 - x2 + units.pixels(-units.nominal_values(o.result.dxvec), pscale=pscale, power=1),
      units.pixels(units.np.linalg.inv(o.result.covariance), pscale=pscale, power=-2),
    )

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
  def layerdict(self):
    return {layer: i for i, layer in enumerate(self.layers)}

  @property
  def overlaps(self): return self.__overlaps
  @property
  def layers(self): return sorted(set.union(*(set(o.layers) for o in self.overlaps)))

  def x(self, layer=None):
    if layer is None:
      return self.__x
    elif hasattr(layer, "layer"):
      return self.x(layer.layer)
    else:
      return self.x()[self.layerdict[layer]]

  def dx(self, overlap):
    return self.x(overlap.layer1) - self.x(overlap.layer2) - (overlap.x1vec - overlap.x2vec)

  @property
  def layerpositions(self):
    result = []
    for layer, row in itertools.zip_longest(self.layers, self.x()):
      x, y = units.nominal_values(row)
      if np.any(units.std_devs(row)):
        (cov_x_x, cov_x_y), (cov_x_y, cov_y_y) = units.covariance_matrix(row)
      else:
        cov_x_x = cov_x_y = cov_y_y = 0
      result.append(
        LayerPosition(
          n=layer,
          x=x,
          y=y,
          cov_x_x=cov_x_x,
          cov_x_y=cov_x_y,
          cov_y_y=cov_y_y,
          pscale=self.pscale,
        )
      )
    return result

  @property
  def layerpositioncovariances(self):
    result = []
    pscale = self.pscale
    for (layer1, row1), (layer2, row2) in itertools.product(itertools.zip_longest(self.layers, self.x()), repeat=2):
      if layer1 == layer2: continue
      if np.any(units.std_devs((*row1, *row2))):
        cov = np.array(units.covariance_matrix((*row1, *row2)))
      else:
        cov = np.zeros(shape=(4, 4))
      result.append(
        LayerPositionCovariance(
          n1=layer1,
          n2=layer2,
          cov_x1_x2=cov[0,2],
          cov_x1_y2=cov[0,3],
          cov_y1_x2=cov[1,2],
          cov_y1_y2=cov[1,3],
          pscale=pscale,
        )
      )
    return result

  def writetable(self, *filenames, check=False, **kwargs):
    positionfilename, covariancefilename = filenames
    writetable(positionfilename, self.layerpositions, **kwargs)
    writetable(covariancefilename, self.layerpositioncovariances, **kwargs)

  def applytooverlaps(self):
    for o in self.overlaps:
      o.stitchresult = self.dx(o)

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

