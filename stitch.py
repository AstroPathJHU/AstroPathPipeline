import dataclasses, itertools, logging, numpy as np, uncertainties as unc, uncertainties.unumpy as unp
from .overlap import OverlapCollection
from .rectangle import Rectangle, RectangleCollection, rectangledict
from .tableio import readtable, writetable
from .utilities import covariance_matrix

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
  A = np.zeros(shape=(size, size))
  b = np.zeros(shape=(size,))
  c = 0

  Txx = -4
  Txy = -3
  Tyx = -2
  Tyy = -1

  rd = rectangledict(rectangles)
  alloverlaps = overlaps
  overlaps = [o for o in overlaps if not o.result.exit]

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
    inversecovariance = np.linalg.inv(o.result.covariance) * scaleby**2 / scaleoverlaperror**2

    A[ii] += inversecovariance / 2
    A[ij] -= inversecovariance / 2
    A[ji] -= inversecovariance / 2
    A[jj] += inversecovariance / 2

    i = np.ix_((ix, iy))
    j = np.ix_((jx, jy))

    constpiece = (-o.result.dxvec - o.x1vec + o.x2vec) / scaleby

    b[i] += inversecovariance @ constpiece
    b[j] -= inversecovariance @ constpiece

    c += constpiece @ inversecovariance @ constpiece / 2

  dxs, dys = zip(*(o.result.dxdy for o in overlaps))

  weightedvariancedx = np.average(
    unp.nominal_values(dxs)**2,
    weights=1/unp.std_devs(dxs)**2,
  )
  sigmax = np.sqrt(weightedvariancedx) / scaleby * scalejittererror

  weightedvariancedy = np.average(
    unp.nominal_values(dys)**2,
    weights=1/unp.std_devs(dys)**2,
  )
  sigmay = np.sqrt(weightedvariancedy) / scaleby * scalejittererror

  if fixpoint == "origin":
    x0vec = np.array([0, 0]) #fix the origin, linear scaling is with respect to that
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

  result = np.linalg.solve(2*A, -b)

  delta2nllfor1sigma = 1

  covariancematrix = np.linalg.inv(A) * delta2nllfor1sigma
  result = np.array(unc.correlated_values(result, covariancematrix))

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

  for o in overlaps:
    x1 = rectanglex[o.p1]
    x2 = rectanglex[o.p2]
    twonll += 0.5 * cp.quad_form(
      x1 - x2 - o.result.dxvec - o.x1vec + o.x2vec,
      np.linalg.inv(o.result.covariance)
    )

  dxs, dys = zip(*(o.result.dxdy for o in overlaps))

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

  def applytooverlaps(self):
    for o in self.overlaps:
      o.stitchresult = self.dx(o)

class StitchResultFullCovariance(StitchResultBase):
  def __init__(self, *, x, T, covariancematrix, **kwargs):
    self.__x = x
    self.T = T
    self.covariancematrix = covariancematrix
    super().__init__(**kwargs)

  @property
  def covariancematrix(self): return self.__covariancematrix
  @covariancematrix.setter
  def covariancematrix(self, matrix):
    if matrix is not None:
      matrix = (matrix + matrix.T) / 2
    self.__covariancematrix = matrix
    self.__covarianceeig = None
  @property
  def covarianceeig(self):
    if self.__covarianceeig is None:
      cov = self.covariancematrix
      if cov is None: raise ValueError("can't get eigenvalues/vectors from matrix because the matrix is None")
      self.__covarianceeig = np.linalg.eig(cov)
    return self.__covarianceeig
  @property
  def covarianceeigenvalues(self): return self.covarianceeig[0]
  @property
  def covarianceeigenvectors(self): return self.covarianceeig[1]

  def sanitycheck(self):
    for thing, errorsq in zip(
      itertools.chain(np.ravel(self.x()), np.ravel(self.T)),
      np.diag(self.covariancematrix)
    ): np.testing.assert_allclose(unc.std_dev(thing)**2, errorsq)

  def x(self, rectangle_or_id=None):
    if rectangle_or_id is None: return self.__x
    if isinstance(rectangle_or_id, Rectangle): rectangle_or_id = rectangle_or_id.n
    return self.__x[self.rectangledict[rectangle_or_id]]

  def dx(self, overlap):
    return self.x(overlap.p1) - self.x(overlap.p2) - (overlap.x1vec - overlap.x2vec)

  def writetable(self, *filenames, neigenvectors=10, trymoreeigenvectors=True, rtol=1e-3, atol=1e-5, **kwargs):
    filename, affinefilename, overlapcovariancefilename, eigenvaluefilename, eigenvectorfilename = filenames

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
          T=self.T
        )
      )
    writetable(filename, rows, **kwargs)

    val, vec = self.covarianceeig
    #sign convention for eigenvectors - doesn't really matter but we want consistent results each time we run
    for v in vec.T:
      if v[0] < 0: v *= -1
    sortidx = val.argsort()[::-1]
    val = val[sortidx]
    vec = vec[:,sortidx]

    nkeep = neigenvectors

    values = []
    for n, v in enumerate(val[:nkeep]):
      values.append(StitchCovarianceEigenvalue(n=n, value=v))
    writetable(eigenvaluefilename, values, rowclass=StitchCovarianceEigenvalue, **kwargs)
    vectors = []
    for n, vector in enumerate(vec.T[:nkeep]):
      for m, entry in enumerate(vector):
        vectors.append(StitchCovarianceEigenvectorEntry(nvector=n, nentry=m, value=entry))
    writetable(eigenvectorfilename, vectors, rowclass=StitchCovarianceEigenvectorEntry, **kwargs)

    overlapcovariances = []
    for o in self.overlaps:
      if o.p2 < o.p1: continue
      covariance = np.array(covariance_matrix(np.concatenate([self.x(o.p1), self.x(o.p2)])))
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
    writetable(overlapcovariancefilename, overlapcovariances, **kwargs)

    try:
      logger.debug("reading back from the file")
      readback = ReadStitchResult(filename, affinefilename, overlapcovariancefilename, eigenvaluefilename, eigenvectorfilename, rectangles=self.rectangles, overlaps=self.overlaps)
      logger.debug("done reading")
      x1 = self.x()
      T1 = self.T
      x2 = readback.x()
      T2 = readback.T
      logger.debug("comparing nominals")
      np.testing.assert_allclose(unp.nominal_values(x1), unp.nominal_values(x2), atol=atol, rtol=rtol)
      np.testing.assert_allclose(unp.nominal_values(T1), unp.nominal_values(T2), atol=atol, rtol=rtol)
      logger.debug("comparing individual errors")
      np.testing.assert_allclose(unp.std_devs(x1), unp.std_devs(x2), atol=atol, rtol=rtol)
      np.testing.assert_allclose(unp.std_devs(T1), unp.std_devs(T2), atol=atol, rtol=rtol)
      logger.debug("comparing overlap errors")
      for o in self.overlaps:
        np.testing.assert_allclose(covariance_matrix(self.dx(o)), covariance_matrix(readback.dx(o)), atol=atol, rtol=rtol)
      logger.debug("done")

      #np.testing.assert_allclose(self.covariancematrix, readback.covariancematrix, atol=atol, rtol=rtol)

    except (BadCovarianceError, AssertionError):
      if not trymoreeigenvectors: raise
      if nkeep >= len(val): raise BadCovarianceError("Inconsistency when writing the covariance matrix to files and reading it back, even when we write all the eigenvectors")
      logger.warning(f"{neigenvectors} eigenvectors (+ field errors and overlap covariances) are not enough to represent the covariance matrix within tolerance, trying with {neigenvectors+1}")
      return self.writetable(*filenames, neigenvectors=neigenvectors+1, **kwargs)

  def readtable(self, *filenames, adjustoverlaps=True):
    filename, affinefilename, overlapcovariancefilename, eigenvaluefilename, eigenvectorfilename = filenames

    coordinates = readtable(filename, StitchCoordinate)
    affines = readtable(affinefilename, AffineEntry)
    overlapcovariances = readtable(overlapcovariancefilename, StitchOverlapCovariance)
    eigenvalues = readtable(eigenvaluefilename, StitchCovarianceEigenvalue)
    eigenvectorentries = readtable(eigenvectorfilename, StitchCovarianceEigenvectorEntry)

    n = len(self.rectangles)
    nominal = np.ndarray(2*n+4)

    eigenvectors = np.zeros((2*n+4, len(eigenvalues)))
    eigenvalues = np.array([v.value for v in sorted(eigenvalues, key=lambda v: v.n)])

    for entry in eigenvectorentries:
      eigenvectors[entry.nentry, entry.nvector] = entry.value

    covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    rd = self.rectangledict

    iTxx = -4
    iTxy = -3
    iTyx = -2
    iTyy = -1

    for coordinate in coordinates:
      ix = 2*rd[coordinate.hpfid]
      iy = ix+1

      nominal[ix] = coordinate.x
      nominal[iy] = coordinate.y
      if adjustoverlaps:
        covariance[ix,ix] = coordinate.cov_x_x
        covariance[ix,iy] = covariance[iy,ix] = coordinate.cov_x_y
        covariance[iy,iy] = coordinate.cov_y_y

    dct = {affine.description: affine.value for affine in affines}

    nominal[iTxx] = dct["axx"]
    nominal[iTxy] = dct["axy"]
    nominal[iTyx] = dct["ayx"]
    nominal[iTyy] = dct["ayy"]

    if adjustoverlaps:
      covariance[iTxx,iTxx] = dct["cov_axx_axx"]
      covariance[iTxx,iTxy] = covariance[iTxy,iTxx] = dct["cov_axx_axy"]
      covariance[iTxx,iTyx] = covariance[iTyx,iTxx] = dct["cov_axx_ayx"]
      covariance[iTxx,iTyy] = covariance[iTyy,iTxx] = dct["cov_axx_ayy"]

      covariance[iTxy,iTxy] = dct["cov_axy_axy"]
      covariance[iTxy,iTyx] = covariance[iTyx,iTxy] = dct["cov_axy_ayx"]
      covariance[iTxy,iTyy] = covariance[iTyy,iTxy] = dct["cov_axy_ayy"]

      covariance[iTyx,iTyx] = dct["cov_ayx_ayx"]
      covariance[iTyx,iTyy] = covariance[iTyy,iTyx] = dct["cov_ayx_ayy"]

      covariance[iTyy,iTyy] = dct["cov_ayy_ayy"]

    rectangledict = self.rectangledict
    for oc in overlapcovariances:
      p1 = oc.hpfid1
      p2 = oc.hpfid2
      ix = 2*rd[p1]
      iy = 2*rd[p1]+1
      jx = 2*rd[p2]
      jy = 2*rd[p2]+1

      if adjustoverlaps:
        covariance[ix,jx] = covariance[jx,ix] = oc.cov_x1_x2
        covariance[ix,jy] = covariance[jy,ix] = oc.cov_x1_y2
        covariance[iy,jx] = covariance[jx,iy] = oc.cov_y1_x2
        covariance[iy,jy] = covariance[jy,iy] = oc.cov_y1_y2

    self.covariancematrix = covariance

    val, vec = self.covarianceeig
    val[::-1].sort()
    if val[-1] < -val[0] * 1e-8:
      raise BadCovarianceError(f"Covariance matrix isn't positive definite: eigenvalues are\n{val}")

    x, T = np.split(
      np.array(
        unc.correlated_values(nominal, covariance)
      ).reshape(n+2, 2),
      [-2],
    )

    self.__x = x
    self.T = T
    self.covariancematrix = covariance

class ReadStitchResult(StitchResultFullCovariance):
  def __init__(self, *args, rectangles, overlaps, **kwargs):
    super().__init__(rectangles=rectangles, overlaps=overlaps, x=None, T=None, covariancematrix=None)
    self.readtable(*args, **kwargs)
    self.sanitycheck()

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
class StitchCoordinate:
  hpfid: int
  x: float
  y: float
  cov_x_x: float
  cov_x_y: float
  cov_y_y: float

def stitchcoordinate(*, position=None, T=None, **kwargs):
  kw2 = {}
  if position is not None:
    kw2["x"], kw2["y"] = unp.nominal_values(position)
    (kw2["cov_x_x"], kw2["cov_x_y"]), (kw2["cov_x_y"], kw2["cov_y_y"]) = covariance_matrix(position)

  return StitchCoordinate(**kwargs, **kw2)

@dataclasses.dataclass
class AffineEntry:
  n: int
  value: float
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
      value = covariance_matrix([entry1.matrixentry, entry2.matrixentry])[0][1]
    super().__init__(n=n, value=value, description = "cov_"+entry1.description+"_"+entry2.description)

  def __post_init__(self): pass

@dataclasses.dataclass
class StitchOverlapCovariance:
  hpfid1: int
  hpfid2: int
  cov_x1_x2: float
  cov_x1_y2: float
  cov_y1_x2: float
  cov_y1_y2: float

@dataclasses.dataclass
class StitchCovarianceEigenvalue:
  n: int
  value: float

@dataclasses.dataclass
class StitchCovarianceEigenvectorEntry:
  nvector: int
  nentry: int
  value: float

class BadCovarianceError(Exception): pass
