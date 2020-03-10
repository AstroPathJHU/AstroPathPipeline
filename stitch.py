import dataclasses, itertools, numpy as np, uncertainties as unc, uncertainties.unumpy as unp
from .rectangle import Rectangle
from .tableio import writetable

def stitch(*, saveresult, usecvxpy=False, **kwargs):
  result = (__stitch_cvxpy if usecvxpy else __stitch)(**kwargs)

  if saveresult:
    result.applytooverlaps()

  return result

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


  rectangledict = {rectangle.n: i for i, rectangle in enumerate(rectangles)}
  alloverlaps = overlaps
  overlaps = [o for o in overlaps if not o.result.exit]

  for o in overlaps:
    ix = 2*rectangledict[o.p1]
    iy = 2*rectangledict[o.p1]+1
    jx = 2*rectangledict[o.p2]
    jy = 2*rectangledict[o.p2]+1
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
    ix = 2*rectangledict[r.n]
    iy = 2*rectangledict[r.n]+1

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

  return StitchResult(x=x, T=T, A=A, b=b, c=c, rectangledict=rectangledict, overlaps=alloverlaps, covariancematrix=covariancematrix)

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

  rectangledict = {rectangle.n: i for i, rectangle in enumerate(rectangles)}
  return StitchResultCvxpy(x=x, T=T, problem=prob, rectangledict=rectangledict, overlaps=alloverlaps)

class StitchResultBase:
  def __init__(self, x, T, rectangledict, overlaps, covariancematrix):
    self.__x = x
    self.T = T
    self.__rectangledict = rectangledict
    self.__overlaps = overlaps
    self.covariancematrix = covariancematrix

    #sanity check
    for thing, errorsq in zip(
      itertools.chain(np.ravel(x), np.ravel(T)),
      np.diag(covariancematrix)
    ): np.testing.assert_allclose(unc.std_dev(thing)**2, errorsq)

  def x(self, rectangle_or_id=None):
    if rectangle_or_id is None: return self.__x
    if isinstance(rectangle_or_id, Rectangle): rectangle_or_id = rectangle_or_id.n
    return self.__x[self.__rectangledict[rectangle_or_id]]

  def dx(self, overlap):
    return self.x(overlap.p1) - self.x(overlap.p2) - (overlap.x1vec - overlap.x2vec)

  def writetable(self, filename, affinefilename, covariancefilename, *, printevery=10000, **kwargs):
    n = 0
    rows = []
    for rectangleid in self.__rectangledict:
      for coordinate, position in enumerate(self.x(rectangleid)):
        n += 1
        rows.append(
          StitchCoordinate(
            n=n,
            hpfid=rectangleid,
            coordinate=coordinate,
            position=position,
          )
        )
    writetable(filename, rows, printevery=printevery, **kwargs)

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
    writetable(affinefilename, affine, printevery=printevery, rowclass=AffineEntry, **kwargs)

    overlapcovariances = []
    for o in self.__overlaps:
      covariance = unc.covariance_matrix(self.dx(o))
      overlapcovariances.append(
        StitchOverlapCovariance(
          hpfid1=o.p1,
          hpfid2=o.p2,
          covxx=covariance[0][0],
          covxy=covariance[0][1],
          covyy=covariance[1][1],
        )
      )
    writetable(covariancefilename, overlapcovariances, printevery=printevery, **kwargs)

  def applytooverlaps(self):
    for o in self.__overlaps:
      o.stitchresult = self.dx(o)

@dataclasses.dataclass
class StitchCoordinate:
  n: int
  hpfid: int
  coordinate: int #for a rectangle: 0=x, 1=y
  position: float

  def __init__(self, n, hpfid, coordinate, position):
    self.n = n
    self.hpfid = hpfid
    self.coordinate = coordinate
    self.positionwithuncertainty = position
    self.position = unc.nominal_value(position)

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
      value = entry1.matrixentry.s
    else:
      value = unc.covariance_matrix([entry1.matrixentry, entry2.matrixentry])[0][1]
    super().__init__(n=n, value=value, description = "cov_"+entry1.description+"_"+entry2.description)

  def __post_init__(self): pass

@dataclasses.dataclass
class StitchOverlapCovariance:
  hpfid1: int
  hpfid2: int
  covxx: float
  covyy: float
  covxy: float

class StitchResult(StitchResultBase):
  def __init__(self, x, T, rectangledict, overlaps, A, b, c, covariancematrix):
    super().__init__(x=x, T=T, rectangledict=rectangledict, overlaps=overlaps, covariancematrix=covariancematrix)
    self.A = A
    self.b = b
    self.c = c

class StitchResultCvxpy(StitchResultBase):
  def __init__(self, x, T, rectangledict, overlaps, problem):
    super().__init__(
      x=x.value,
      T=T.value,
      rectangledict=rectangledict,
      overlaps=overlaps,
      covariancematrix=np.zeros(x.size+T.size, x.size+T.size)
    )
    self.problem = problem
    self.xvar = x
    self.Tvar = T

