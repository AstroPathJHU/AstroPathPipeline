import dataclasses, itertools, numpy as np, uncertainties as unc, uncertainties.unumpy as unp
from .rectangle import Rectangle
from .tableio import readtable, writetable

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

class StitchResultBase:
  def __init__(self, *, x, T, rectangles, overlaps, covariancematrix):
    self.__x = x
    self.T = T
    self.rectangles = rectangles
    self.__overlaps = overlaps
    self.covariancematrix = covariancematrix

    #sanity check
    for thing, errorsq in zip(
      itertools.chain(np.ravel(x), np.ravel(T)),
      np.diag(covariancematrix)
    ): np.testing.assert_allclose(unc.std_dev(thing)**2, errorsq)

  @property
  def rectangledict(self):
    return rectangledict(self.rectangles)

  def x(self, rectangle_or_id=None):
    if rectangle_or_id is None: return self.__x
    if isinstance(rectangle_or_id, Rectangle): rectangle_or_id = rectangle_or_id.n
    return self.__x[self.rectangledict[rectangle_or_id]]

  def dx(self, overlap):
    return self.x(overlap.p1) - self.x(overlap.p2) - (overlap.x1vec - overlap.x2vec)

  def writetable(self, filename, affinefilename, covariancefilename, *, printevery=10000, **kwargs):
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

    rows = []
    for rectangleid in self.rectangledict:
      rows.append(
        stitchcoordinate(
          hpfid=rectangleid,
          position=self.x(rectangleid),
          T=self.T
        )
      )
    writetable(filename, rows, printevery=printevery, **kwargs)

    overlapcovariances = []
    for o in self.__overlaps:
      if o.p2 < o.p1: continue
      covariance = np.array(unc.covariance_matrix(np.concatenate([self.x(o.p1), self.x(o.p2)])))
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
    writetable(covariancefilename, overlapcovariances, printevery=printevery, **kwargs)

  def readtable(self, filename, affinefilename, covariancefilename):
    coordinates = readtable(filename, StitchCoordinate)
    affines = readtable(affinefilename, AffineEntry)
    overlapcovariances = readtable(covariancefilename, StitchOverlapCovariance)

    n = len(self.rectangles)
    rd = self.rectangledict
    nominal = np.ndarray(2*n+4)
    covariance = np.ndarray((2*n+4, 2*n+4))

    iTxx = -4
    iTxy = -3
    iTyx = -2
    iTyy = -1

    for coordinate in coordinates:
      ix = 2*rd[coordinate.hpfid]
      iy = ix+1

      nominal[ix] = coordinate.x
      nominal[iy] = coordinate.y
      covariance[ix,ix] = coordinate.cov_x_x
      covariance[ix,iy] = coordinate.cov_x_y
      covariance[iy,iy] = coordinate.cov_x_y
      covariance[ix,iTxx] = coordinate.cov_x_axx
      covariance[ix,iTxy] = coordinate.cov_x_axy
      covariance[ix,iTyx] = coordinate.cov_x_ayx
      covariance[ix,iTyy] = coordinate.cov_x_ayy
      covariance[iy,iTxx] = coordinate.cov_y_axx
      covariance[iy,iTxy] = coordinate.cov_y_axy
      covariance[iy,iTyx] = coordinate.cov_y_ayx
      covariance[iy,iTyy] = coordinate.cov_y_ayy

    dct = {affine.description: affine.value for affine in affines}

    nominal[iTxx] = dct["axx"]
    nominal[iTxy] = dct["axy"]
    nominal[iTyx] = dct["ayx"]
    nominal[iTyy] = dct["ayy"]

    covariance[iTxx,iTxx] = dct["cov_axx_axx"]
    covariance[iTxx,iTxy] = dct["cov_axx_axy"]
    covariance[iTxx,iTyx] = dct["cov_axx_ayx"]
    covariance[iTxx,iTyy] = dct["cov_axx_ayy"]

    covariance[iTxy,iTxy] = dct["cov_axy_axy"]
    covariance[iTxy,iTyx] = dct["cov_axy_ayx"]
    covariance[iTxy,iTyy] = dct["cov_axy_ayy"]

    covariance[iTyx,iTyx] = dct["cov_ayx_ayx"]
    covariance[iTyx,iTyy] = dct["cov_ayx_ayy"]

    covariance[iTyy,iTyy] = dct["cov_ayy_ayy"]

    rectangledict = self.rectangledict
    for oc in overlapcovariances:
      p1 = oc.hpfid1
      p2 = oc.hpfid2
      ix = 2*rd[p1]
      iy = 2*rd[p1]+1
      jx = 2*rd[p2]
      jy = 2*rd[p2]+1

      covariancematrix[ix,jx] -= oc.cov_x1_x2
      covariancematrix[ix,jy] -= oc.cov_x1_y2
      covariancematrix[iy,jx] -= oc.cov_y1_x2
      covariancematrix[iy,jy] -= oc.cov_y1_y2

    covariancematrix += covariancematrix.T - np.diag(np.diag(covariancematrix))
    self.covariancematrix = covariancematrix

    x, T = np.split(
      np.array(
        unc.correlated_values(covariancematrix)
      ).reshape(self.nrectangles+2, 2),
      [-2],
    )

    self.__x = x
    self.T = T
    self.covariancematrix = covariancematrix

  def applytooverlaps(self):
    for o in self.__overlaps:
      o.stitchresult = self.dx(o)

@dataclasses.dataclass
class StitchCoordinate:
  hpfid: int
  x: float
  y: float
  cov_x_x: float
  cov_x_y: float
  cov_y_y: float
  cov_x_axx: float
  cov_x_axy: float
  cov_x_ayx: float
  cov_x_ayy: float
  cov_y_axx: float
  cov_y_axy: float
  cov_y_ayx: float
  cov_y_ayy: float

def stitchcoordinate(*, position=None, T=None, **kwargs):
  kw2 = {}
  if position is not None:
    kw2["x"], kw2["y"] = unp.nominal_values(position)
    (kw2["cov_x_x"], kw2["cov_x_y"]), (kw2["cov_x_y"], kw2["cov_y_y"]) = unc.covariance_matrix(position)
    if T is not None:
      cov = np.array(unc.covariance_matrix(np.concatenate([position, np.ravel(T)])))
      kw2["cov_x_axx"], kw2["cov_x_axy"], kw2["cov_x_ayx"], kw2["cov_x_ayy"] = cov[0, 2:]
      kw2["cov_y_axx"], kw2["cov_y_axy"], kw2["cov_y_ayx"], kw2["cov_y_ayy"] = cov[1, 2:]

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
      value = unc.covariance_matrix([entry1.matrixentry, entry2.matrixentry])[0][1]
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

class StitchResult(StitchResultBase):
  def __init__(self, *, A, b, c, **kwargs):
    super().__init__(**kwargs)
    self.A = A
    self.b = b
    self.c = c

class StitchResultCvxpy(StitchResultBase):
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

def rectangledict(rectangles):
  return {rectangle.n: i for i, rectangle in enumerate(rectangles)}
