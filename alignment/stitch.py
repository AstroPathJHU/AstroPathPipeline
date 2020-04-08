import abc, dataclasses, itertools, logging, numpy as np, uncertainties as unc, uncertainties.unumpy as unp
from .overlap import OverlapCollection
from .rectangle import Rectangle, RectangleCollection, rectangledict
from ..utilities import units
from ..utilities.tableio import readtable, writetable
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

logger = logging.getLogger("align")

def stitch(*, usecvxpy=False, **kwargs):
  return (__stitch_cvxpy if usecvxpy else __stitch)(**kwargs)

def __stitch(*, rectangles, overlaps, scaleby=1, scalejittererror=1, scaleoverlaperror=1, fixpoint="origin"):
  """
  \begin{align}
  -2 \ln L =&
    1/2 \sum_\text{overlaps}
    (\vec{x}_{p1} - \vec{x}_{p2} - d\vec{x} - \vec{x}_{p1}^n + \vec{x}_{p2}^n)^T \\
    &\mathbf{cov}^{-1}
    (\vec{x}_{p1} - \vec{x}_{p2} - d\vec{x} - \vec{x}_{p1}^n + \vec{x}_{p2}^n) \\
    +&
    \sum_p \left(
      \frac{(\vec{x}_p-\vec{x}_0) - \mathbf{T}(\vec{x}_p^n-\vec{x}_0)}
      {\sigma}
    \right)^2
  \end{align}
  \begin{equation}
  \mathbf{T} = \begin{pmatrix}
  T_{xx} & T_{xy} \\ T_{yx} & T_{yy}
  \end{pmatrix}
  \end{equation}
  """
  #nll = x^T A x + bx + c

  size = 2*len(rectangles) + 4 #2* because each rectangle has an x and a y, + 4 for the components of T
  A = np.zeros(shape=(size, size), dtype=object)
  b = np.zeros(shape=(size,), dtype=object)
  c = 0

  Txx = -4
  Txy = -3
  Tyx = -2
  Tyy = -1

  rd = rectangledict(rectangles)
  alloverlaps = overlaps
  overlaps = [o for o in overlaps if not o.result.exit]
  for o in overlaps[:]:
    if o.p2 > o.p1 and any((oo.p2, oo.p1) == (o.p1, o.p2) for oo in overlaps):
      overlaps.remove(o)

  for o in overlaps:
    ix = 2*rd[o.p1]
    iy = 2*rd[o.p1]+1
    jx = 2*rd[o.p2]
    jy = 2*rd[o.p2]+1
    assert ix >= 0, ix
    assert iy < 2*len(rectangles), iy
    assert jx >= 0, jx
    assert jy < 2*len(rectangles), jy

    ii = np.ix_((ix,iy), (ix,iy))
    ij = np.ix_((ix,iy), (jx,jy))
    ji = np.ix_((jx,jy), (ix,iy))
    jj = np.ix_((jx,jy), (jx,jy))
    inversecovariance = units.linalg.inv(o.result.covariance) * scaleby**2 / scaleoverlaperror**2

    A[ii] += inversecovariance
    A[ij] -= inversecovariance
    A[ji] -= inversecovariance
    A[jj] += inversecovariance

    i = np.ix_((ix, iy))
    j = np.ix_((jx, jy))

    constpiece = (-units.nominal_values(o.result.dxvec) - o.x1vec + o.x2vec) / scaleby

    b[i] += 2 * inversecovariance @ constpiece
    b[j] -= 2 * inversecovariance @ constpiece

    c += constpiece @ inversecovariance @ constpiece

  dxs, dys = zip(*(o.result.dxvec for o in overlaps))

  weightedvariancedx = np.average(
    units.nominal_values(dxs)**2,
    weights=1/units.std_devs(dxs)**2,
  )
  sigmax = np.sqrt(weightedvariancedx) / scaleby * scalejittererror

  weightedvariancedy = np.average(
    units.nominal_values(dys)**2,
    weights=1/units.std_devs(dys)**2,
  )
  sigmay = np.sqrt(weightedvariancedy) / scaleby * scalejittererror

  if fixpoint == "origin":
    x0vec = units.distances(pixels=np.array([0, 0]), pscale=c.pscale) #fix the origin, linear scaling is with respect to that
  elif fixpoint == "center":
    x0vec = np.mean([r.xvec for r in rectangles], axis=0)
  else:
    x0vec = fixpoint

  x0, y0 = x0vec

  for r in rectangles:
    ix = 2*rd[r.n]
    iy = 2*rd[r.n]+1

    x = r.x / scaleby
    y = r.y / scaleby

    A[ix,ix] += 1 / sigmax**2
    A[iy,iy] += 1 / sigmay**2
    A[(ix, Txx), (Txx, ix)] -= (x - x0) / sigmax**2
    A[(ix, Txy), (Txy, ix)] -= (y - y0) / sigmax**2
    A[(iy, Tyx), (Tyx, iy)] -= (x - x0) / sigmay**2
    A[(iy, Tyy), (Tyy, iy)] -= (y - y0) / sigmay**2

    A[Txx, Txx]               += (x - x0)**2       / sigmax**2
    A[(Txx, Txy), (Txy, Txx)] += (x - x0)*(y - y0) / sigmax**2
    A[Txy, Txy]               += (y - y0)**2       / sigmax**2

    A[Tyx, Tyx]               += (x - x0)**2       / sigmay**2
    A[(Tyx, Tyy), (Tyy, Tyx)] += (x - x0)*(y - y0) / sigmay**2
    A[Tyy, Tyy]               += (y - y0)**2       / sigmay**2

    b[ix] -= 2 * x0 / sigmax**2
    b[iy] -= 2 * y0 / sigmay**2

    b[Txx] += 2 * x0 * (x - x0) / sigmax**2
    b[Txy] += 2 * x0 * (y - y0) / sigmax**2
    b[Tyx] += 2 * y0 * (x - x0) / sigmay**2
    b[Tyy] += 2 * y0 * (y - y0) / sigmay**2

    c += x0**2 / sigmax**2
    c += y0**2 / sigmay**2

  result = units.linalg.solve(2*A, -b)

  delta2nllfor1sigma = 1

  covariancematrix = units.linalg.inv(A) * delta2nllfor1sigma
  result = np.array(units.correlated_distances(distances=result, covariance=covariancematrix))

  x = result[:-4].reshape(len(rectangles), 2) * scaleby
  T = result[-4:].reshape(2, 2)

  return StitchResult(x=x, T=T, A=A, b=b, c=c, rectangles=rectangles, overlaps=alloverlaps, covariancematrix=covariancematrix)

def __stitch_cvxpy(*, overlaps, rectangles, fixpoint="origin"):
  """
  \begin{align}
  -2 \ln L =&
    1/2 \sum_\text{overlaps}
    (\vec{x}_{p1} - \vec{x}_{p2} - d\vec{x} - \vec{x}_{p1}^n + \vec{x}_{p2}^n)^T \\
    &\mathbf{cov}^{-1}
    (\vec{x}_{p1} - \vec{x}_{p2} - d\vec{x} - \vec{x}_{p1}^n + \vec{x}_{p2}^n) \\
    +&
    \sum_p \left(
      \frac{(\vec{x}_p-\vec{x}_0) - \mathbf{T}(\vec{x}_p^n-\vec{x}_0)}
      {\sigma}
    \right)^2
  \end{align}
  \begin{equation}
  \mathbf{T} = \begin{pmatrix}
  T_{xx} & T_{xy} \\ T_{yx} & T_{yy}
  \end{pmatrix}
  \end{equation}
  """

  try:
    import cvxpy as cp
  except ImportError:
    raise ImportError("To stitch with cvxpy, you have to install cvxpy")

  x = cp.Variable(shape=(len(rectangles), 2))
  T = cp.Variable(shape=(2, 2))

  twonll = 0
  rectanglex = {r.n: xx for r, xx in zip(rectangles, x)}
  alloverlaps = overlaps
  overlaps = [o for o in overlaps if not o.result.exit]
  for o in overlaps[:]:
    if o.p2 > o.p1 and any((oo.p2, oo.p1) == (o.p1, o.p2) for oo in overlaps):
      overlaps.remove(o)

  for o in overlaps:
    x1 = rectanglex[o.p1]
    x2 = rectanglex[o.p2]
    twonll += cp.quad_form(
      x1 - x2 - unp.nominal_values(o.result.dxvec) - o.x1vec + o.x2vec,
      np.linalg.inv(o.result.covariance)
    )

  dxs, dys = zip(*(o.result.dxvec for o in overlaps))

  weightedvariancedx = np.average(
    unp.nominal_values(dxs)**2,
    weights=1/unp.std_devs(dxs)**2,
  )
  sigmax = np.sqrt(weightedvariancedx)

  weightedvariancedy = np.average(
    unp.nominal_values(dys)**2,
    weights=1/unp.std_devs(dys)**2,
  )
  sigmay = np.sqrt(weightedvariancedy)

  sigma = np.array(sigmax, sigmay)

  if fixpoint == "origin":
    x0vec = np.array([0, 0]) #fix the origin, linear scaling is with respect to that
  elif fixpoint == "center":
    x0vec = np.mean([r.xvec for r in rectangles], axis=0)
  else:
    x0vec = fixpoint

  for r in rectangles:
    twonll += cp.norm(((rectanglex[r.n] - x0vec) - T @ (r.xvec - x0vec)) / sigma)

  minimize = cp.Minimize(twonll)
  prob = cp.Problem(minimize)
  prob.solve()

  return StitchResultCvxpy(x=x, T=T, problem=prob, rectangles=rectangles, overlaps=alloverlaps)

class StitchResultBase(OverlapCollection, RectangleCollection):
  def __init__(self, *, rectangles, overlaps):
    self.__rectangles = rectangles
    self.__overlaps = overlaps

  @property
  def overlaps(self): return self.__overlaps
  @property
  def rectangles(self): return self.__rectangles

  @abc.abstractmethod
  def x(self, rectangle_or_id=None): pass
  @abc.abstractmethod
  def dx(self, overlap): pass
  @abc.abstractproperty
  def T(self): pass
  @abc.abstractproperty
  def overlapcovariances(self): pass

  def applytooverlaps(self):
    for o in self.overlaps:
      o.stitchresult = self.dx(o)

  def writetable(self, *filenames, rtol=1e-3, atol=1e-5, check=False, **kwargs):
    filename, affinefilename, overlapcovariancefilename = filenames

    affine = []
    n = 0
    for rowcoordinate, row in zip("xy", self.T):
      for columncoordinate, entry in zip("xy", row):
        n += 1
        affine.append(
          AffineNominalEntry(
            n=n,
            matrixentry=entry,
            description="a"+rowcoordinate+columncoordinate
          )
        )

    for entry1, entry2 in itertools.combinations_with_replacement(affine[:], 2):
      n += 1
      affine.append(AffineCovarianceEntry(n=n, entry1=entry1, entry2=entry2))
    writetable(affinefilename, affine, rowclass=AffineEntry, **kwargs)

    rows = []
    for rectangleid in self.rectangledict:
      rows.append(
        stitchcoordinate(
          hpfid=rectangleid,
          position=self.x(rectangleid),
        )
      )
    writetable(filename, rows, **kwargs)

    writetable(overlapcovariancefilename, self.overlapcovariances, **kwargs)

    if check:
      logger.debug("reading back from the file")
      readback = ReadStitchResult(*filenames, rectangles=self.rectangles, overlaps=self.overlaps)
      logger.debug("done reading")
      x1 = self.x()
      T1 = self.T
      x2 = readback.x()
      T2 = readback.T
      logger.debug("comparing nominals")
      units.testing.assert_allclose(units.nominal_values(x1), units.nominal_values(x2), atol=atol, rtol=rtol)
      units.testing.assert_allclose(units.nominal_values(T1), units.nominal_values(T2), atol=atol, rtol=rtol)
      logger.debug("comparing individual errors")
      units.testing.assert_allclose(units.std_devs(x1), units.std_devs(x2), atol=atol, rtol=rtol)
      units.testing.assert_allclose(units.std_devs(T1), units.std_devs(T2), atol=atol, rtol=rtol)
      logger.debug("comparing overlap errors")
      for o in self.overlaps:
        units.testing.assert_allclose(units.covariance_matrix(self.dx(o)), units.covariance_matrix(readback.dx(o)), atol=atol, rtol=rtol)
      logger.debug("done")

class StitchResultFullCovariance(StitchResultBase):
  def __init__(self, *, x, T, covariancematrix, **kwargs):
    self.__x = x
    self.__T = T
    self.covariancematrix = covariancematrix
    super().__init__(**kwargs)

  @property
  def T(self): return self.__T

  @property
  def covariancematrix(self): return self.__covariancematrix
  @covariancematrix.setter
  def covariancematrix(self, matrix):
    if matrix is not None:
      matrix = (matrix + matrix.T) / 2
    self.__covariancematrix = matrix

  def sanitycheck(self):
    for thing, errorsq in zip(
      itertools.chain(np.ravel(self.x()), np.ravel(self.T)),
      np.diag(self.covariancematrix)
    ): units.testing.assert_allclose(units.std_dev(thing)**2, errorsq)

  def x(self, rectangle_or_id=None):
    if rectangle_or_id is None: return self.__x
    if isinstance(rectangle_or_id, Rectangle): rectangle_or_id = rectangle_or_id.n
    return self.__x[self.rectangledict[rectangle_or_id]]

  def dx(self, overlap):
    return self.x(overlap.p1) - self.x(overlap.p2) - (overlap.x1vec - overlap.x2vec)

  @property
  def overlapcovariances(self):
    overlapcovariances = []
    for o in self.overlaps:
      if o.p2 < o.p1: continue
      covariance = np.array(units.covariance_matrix(np.concatenate([self.x(o.p1), self.x(o.p2)])))
      overlapcovariances.append(
        StitchOverlapCovariance(
          hpfid1=o.p1,
          hpfid2=o.p2,
          cov_x1_x2=covariance[0,2],
          cov_x1_y2=covariance[0,3],
          cov_y1_x2=covariance[1,2],
          cov_y1_y2=covariance[1,3],
        )
      )
    return overlapcovariances

class StitchResultOverlapCovariances(StitchResultBase):
  def __init__(self, *, x, T, overlapcovariances, **kwargs):
    self.__x = x
    self.__T = T
    self.__overlapcovariances = overlapcovariances
    super().__init__(**kwargs)

  @property
  def T(self): return self.__T
  @property
  def overlapcovariances(self): return self.__overlapcovariances

  def x(self, rectangle_or_id=None):
    if rectangle_or_id is None: return self.__x
    if isinstance(rectangle_or_id, Rectangle): rectangle_or_id = rectangle_or_id.n
    return self.__x[self.rectangledict[rectangle_or_id]]

  def dx(self, overlap):
    x1 = self.x(overlap.p1)
    x2 = self.x(overlap.p2)
    overlapcovariance = self.overlapcovariance(overlap)

    nominals = np.concatenate([units.nominal_values(x1), units.nominal_values(x2)])

    covariance = np.zeros((4, 4), dtype=object)
    covariance[:2,:2] = units.covariance_matrix(x1)
    covariance[2:,2:] = units.covariance_matrix(x2)
    covariance[0,2] = covariance[2,0] = overlapcovariance.cov_x1_x2
    covariance[0,3] = covariance[3,0] = overlapcovariance.cov_x1_y2
    covariance[1,2] = covariance[2,1] = overlapcovariance.cov_y1_x2
    covariance[1,3] = covariance[3,1] = overlapcovariance.cov_y1_y2

    xx1, yy1, xx2, yy2 = units.correlated_distances(distances=nominals, covariance=covariance)
    newx1 = np.array([xx1, yy1])
    newx2 = np.array([xx2, yy2])

    units.testing.assert_allclose(units.nominal_values(x1), units.nominal_values(newx1))
    units.testing.assert_allclose(units.covariance_matrix(x1), units.covariance_matrix(newx1))
    units.testing.assert_allclose(units.nominal_values(x2), units.nominal_values(newx2))
    units.testing.assert_allclose(unc.covariance_matrix(x2), units.covariance_matrix(newx2))

    return newx1 - newx2 - (overlap.x1vec - overlap.x2vec)

  def overlapcovariance(self, overlap):
    for oc in self.overlapcovariances:
      if {overlap.p1, overlap.p2} == {oc.hpfid1, oc.hpfid2}:
        return oc
    raise KeyError(f"No overlap covariance with {overlap.p1} {overlap.p2}")

  def readtable(self, *filenames, adjustoverlaps=True):
    filename, affinefilename, overlapcovariancefilename = filenames

    pscale = {_.pscale for _ in itertools.chain(self.rectangles, self.overlaps)}
    if len(pscale) != 1: raise ValueError("?????? this should never happen")
    pscale = pscale.pop()

    coordinates = readtable(filename, StitchCoordinate, extrakwargs={"pscale": pscale})
    affines = readtable(affinefilename, AffineEntry)
    overlapcovariances = readtable(overlapcovariancefilename, StitchOverlapCovariance, extrakwargs={"pscale": pscale})

    self.__x = np.array([coordinate.xvec for coordinate in coordinates])
    self.__overlapcovariances = overlapcovariances

    iTxx, iTxy, iTyx, iTyy = range(4)

    dct = {affine.description: affine.value for affine in affines}

    Tnominal = np.ndarray(4)
    Tcovariance = np.ndarray((4, 4))

    Tnominal[iTxx] = dct["axx"]
    Tnominal[iTxy] = dct["axy"]
    Tnominal[iTyx] = dct["ayx"]
    Tnominal[iTyy] = dct["ayy"]

    Tcovariance[iTxx,iTxx] = dct["cov_axx_axx"]
    Tcovariance[iTxx,iTxy] = Tcovariance[iTxy,iTxx] = dct["cov_axx_axy"]
    Tcovariance[iTxx,iTyx] = Tcovariance[iTyx,iTxx] = dct["cov_axx_ayx"]
    Tcovariance[iTxx,iTyy] = Tcovariance[iTyy,iTxx] = dct["cov_axx_ayy"]

    Tcovariance[iTxy,iTxy] = dct["cov_axy_axy"]
    Tcovariance[iTxy,iTyx] = Tcovariance[iTyx,iTxy] = dct["cov_axy_ayx"]
    Tcovariance[iTxy,iTyy] = Tcovariance[iTyy,iTxy] = dct["cov_axy_ayy"]

    Tcovariance[iTyx,iTyx] = dct["cov_ayx_ayx"]
    Tcovariance[iTyx,iTyy] = Tcovariance[iTyy,iTyx] = dct["cov_ayx_ayy"]

    Tcovariance[iTyy,iTyy] = dct["cov_ayy_ayy"]

    self.__T = np.array(unc.correlated_values(Tnominal, Tcovariance)).reshape((2, 2))

class ReadStitchResult(StitchResultOverlapCovariances):
  def __init__(self, *args, rectangles, overlaps, **kwargs):
    super().__init__(rectangles=rectangles, overlaps=overlaps, x=None, T=None, overlapcovariances=None)
    self.readtable(*args, **kwargs)

class CalculatedStitchResult(StitchResultFullCovariance):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sanitycheck()

class StitchResult(CalculatedStitchResult):
  def __init__(self, *, A, b, c, **kwargs):
    super().__init__(**kwargs)
    self.A = A
    self.b = b
    self.c = c

class StitchResultCvxpy(CalculatedStitchResult):
  def __init__(self, *, x, T, problem, **kwargs):
    super().__init__(
      x=x.value,
      T=T.value,
      covariancematrix=np.zeros(x.size+T.size, x.size+T.size),
      **kwargs
    )
    self.problem = problem
    self.xvar = x
    self.Tvar = T

@dataclasses.dataclass
class StitchCoordinate(DataClassWithDistances):
  pixelsormicrons = "pixels"

  hpfid: int
  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  cov_x_x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  cov_x_y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  cov_y_y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  pscale: dataclasses.InitVar[float] = None

  def __post_init__(self, pscale):
    super().__post_init__(pscale=pscale)

    nominal = [self.x, self.y]
    covariance = [[self.cov_x_x, self.cov_x_y], [self.cov_x_y, self.cov_y_y]]
    self.xvec = units.correlated_distances(distances=nominal, covariance=covariance)

print(StitchCoordinate.pscale)

def stitchcoordinate(*, position=None, **kwargs):
  kw2 = {}
  if position is not None:
    kw2["x"], kw2["y"] = units.nominal_values(position)
    (kw2["cov_x_x"], kw2["cov_x_y"]), (kw2["cov_x_y"], kw2["cov_y_y"]) = units.covariance_matrix(position)

  return StitchCoordinate(**kwargs, **kw2)

@dataclasses.dataclass
class AffineEntry:
  n: int
  value: float = dataclasses.field(metadata={"writefunction": float})
  description: str

class AffineNominalEntry(AffineEntry):
  def __init__(self, n, matrixentry, description):
    self.matrixentry = matrixentry
    super().__init__(n=n, value=matrixentry.n, description=description)

  def __post_init__(self): pass

class AffineCovarianceEntry(AffineEntry):
  def __init__(self, n, entry1, entry2):
    self.entry1 = entry1
    self.entry2 = entry2
    if entry1 is entry2:
      value = entry1.matrixentry.s**2
    else:
      value = unc.covariance_matrix([entry1.matrixentry, entry2.matrixentry])[0][1]
    super().__init__(n=n, value=value, description = "cov_"+entry1.description+"_"+entry2.description)

  def __post_init__(self): pass

@dataclasses.dataclass
class StitchOverlapCovariance(DataClassWithDistances):
  pixelsormicrons = "pixels"

  hpfid1: int
  hpfid2: int
  cov_x1_x2: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  cov_x1_y2: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  cov_y1_x2: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  cov_y1_y2: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=2)
  pscale: dataclasses.InitVar[float] = None
