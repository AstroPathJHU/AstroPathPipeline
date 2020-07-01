import abc, collections, dataclasses, itertools, methodtools, more_itertools, numpy as np, uncertainties as unc
from ..prepdb.overlap import RectangleOverlapCollection
from ..prepdb.rectangle import Rectangle, rectangledict
from ..utilities import units
from ..utilities.misc import dummylogger, weightedstd
from ..utilities.tableio import readtable, writetable
from .field import Field, FieldOverlap

def stitch(*, usecvxpy=False, **kwargs):
  return (__stitch_cvxpy if usecvxpy else __stitch)(**kwargs)

def __stitch(*, rectangles, overlaps, scaleby=1, scalejittererror=1, scaleoverlaperror=1, fixpoint="origin", origin=np.array([0, 0]), logger=dummylogger):
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
  logger.info("stitch")

  #nll = x^T A x + bx + c

  size = 2*len(rectangles) + 4 #2* because each rectangle has an x and a y, + 4 for the components of T
  A = np.zeros(shape=(size, size), dtype=units.unitdtype)
  b = np.zeros(shape=(size,), dtype=units.unitdtype)
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
    inversecovariance = units.np.linalg.inv(o.result.covariance) * scaleby**2 / scaleoverlaperror**2

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

  sigmax = weightedstd(dxs, subtractaverage=False) / scaleby * scalejittererror
  sigmay = weightedstd(dys, subtractaverage=False) / scaleby * scalejittererror

  if fixpoint == "origin":
    x0vec = origin
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

  logger.debug("assembled A b c")

  result = units.np.linalg.solve(2*A, -b)

  logger.debug("solved quadratic equation")

  delta2nllfor1sigma = 1

  covariancematrix = units.np.linalg.inv(A) * delta2nllfor1sigma
  logger.debug("got covariance matrix")
  result = np.array(units.correlated_distances(distances=result, covariance=covariancematrix))

  x = result[:-4].reshape(len(rectangles), 2) * scaleby
  T = result[-4:].reshape(2, 2)

  logger.debug("done")

  return StitchResult(x=x, T=T, A=A, b=b, c=c, rectangles=rectangles, overlaps=alloverlaps, covariancematrix=covariancematrix, origin=origin)

def __stitch_cvxpy(*, overlaps, rectangles, fixpoint="origin", origin=np.array([0, 0]), logger=dummylogger):
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

  pscale = {_.pscale for _ in itertools.chain(overlaps, rectangles)}
  if len(pscale) > 1: raise units.UnitsError("Inconsistent pscales")
  pscale = pscale.pop()

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
      x1 - x2 + units.pixels(-units.nominal_values(o.result.dxvec) - o.x1vec + o.x2vec, pscale=pscale, power=1),
      units.pixels(units.np.linalg.inv(o.result.covariance), pscale=pscale, power=-2)
    )

  dxs, dys = zip(*(o.result.dxvec for o in overlaps))

  weightedvariancedx = np.average(
    units.nominal_values(dxs)**2,
    weights=1/units.std_devs(dxs)**2,
  )
  sigmax = np.sqrt(weightedvariancedx)

  weightedvariancedy = np.average(
    units.nominal_values(dys)**2,
    weights=1/units.std_devs(dys)**2,
  )
  sigmay = np.sqrt(weightedvariancedy)

  sigma = np.array([sigmax, sigmay])

  if fixpoint == "origin":
    x0vec = origin
  elif fixpoint == "center":
    x0vec = np.mean([r.xvec for r in rectangles], axis=0)
  else:
    x0vec = fixpoint

  for r in rectangles:
    twonll += cp.norm(
      (
        (rectanglex[r.n] - units.pixels(x0vec, pscale=pscale, power=1))
        - T @ units.pixels(
          (r.xvec - x0vec),
          pscale=pscale, power=1
        )
      ) / units.pixels(sigma, pscale=pscale, power=1)
    )

  minimize = cp.Minimize(twonll)
  prob = cp.Problem(minimize)
  prob.solve()

  return StitchResultCvxpy(x=x, T=T, problem=prob, rectangles=rectangles, overlaps=alloverlaps, pscale=pscale, origin=origin)

class StitchResultBase(RectangleOverlapCollection):
  def __init__(self, *, rectangles, overlaps, origin):
    self.__rectangles = rectangles
    self.__overlaps = overlaps
    self.__origin = origin

  @property
  def pscale(self):
    pscale = {_.pscale for _ in itertools.chain(self.rectangles, self.overlaps)}
    if len(pscale) != 1: raise ValueError("?????? this should never happen")
    return pscale.pop()

  @property
  def overlaps(self): return self.__overlaps
  @property
  def rectangles(self): return self.__rectangles
  @property
  def origin(self): return self.__origin

  @abc.abstractmethod
  def x(self, rectangle_or_id=None): pass
  @abc.abstractmethod
  def dx(self, overlap): pass
  @abc.abstractproperty
  def T(self): pass
  @abc.abstractproperty
  def fieldoverlaps(self): pass

  def applytooverlaps(self):
    for o in self.overlaps:
      o.stitchresult = self.dx(o)

  @methodtools.lru_cache()
  def __fields(self):
    result = []
    islands = list(self.islands(useexitstatus=True))
    gxdict = collections.defaultdict(dict)
    gydict = collections.defaultdict(dict)
    primaryregionsx = {}
    primaryregionsy = {}

    shape = {tuple(r.shape) for r in self.rectangles}
    if len(shape) > 1:
      raise ValueError("Some rectangles have different shapes")
    shape = shape.pop()

    for gc, island in enumerate(islands, start=1):
      rectangles = [self.rectangles[self.rectangledict[n]] for n in island]

      for i, (primaryregions, gdict) in enumerate(zip((primaryregionsx, primaryregionsy), (gxdict, gydict))):
        average = []
        cs = sorted({r.cxvec[i] for r in rectangles})
        for g, c in enumerate(cs, start=1):
          gdict[gc][c] = g
          theserectangles = [r for r in rectangles if r.cxvec[i] == c]
          average.append(np.mean(units.nominal_values([self.x(r)[i] for r in theserectangles])))
        primaryregions[gc] = [(x1+shape[i] + x2)/2 for x1, x2 in more_itertools.pairwise(average)]

        if len(primaryregions[gc]) >= 2:
          m, b = units.np.polyfit(
            x=range(1, len(average)),
            y=primaryregions[gc],
            deg=1,
          )
          primaryregions[gc].insert(0, m*0+b)
          primaryregions[gc].append(m*len(average)+b)
        else:
          allcs = sorted({r.cxvec[i] for r in self.rectangles})
          mindiff = min(np.diff(allcs))
          divideby = 1
          while mindiff / divideby > shape[i]:
            divideby += 1
          mindiff /= divideby

          if len(primaryregions[gc]) == 1:
            primaryregions[gc].insert(0, primaryregions[gc][0] - mindiff)
            primaryregions[gc].append(primaryregions[gc][1] + mindiff)
          else: #len(primaryregions) == 0
            primaryregions[gc].append(average[0] + (shape[i] - mindiff) / 2)
            primaryregions[gc].append(average[0] + (shape[i] + mindiff) / 2)

    for rectangle in self.rectangles:
      for gc, island in enumerate(islands, start=1):
        if rectangle.n in island:
          break
      else:
        assert False
      gx = gxdict[gc][rectangle.cx]
      gy = gydict[gc][rectangle.cy]
      result.append(
        Field(
          rectangle=rectangle,
          ixvec=units.distances(pixels=units.pixels(rectangle.xvec, pscale=self.pscale).round().astype(int), pscale=self.pscale),
          gc=gc,
          pxvec=self.x(rectangle) - self.origin,
          gxvec=(gx, gy),
          primaryregionx=(primaryregionsx[gc][gx-1], primaryregionsx[gc][gx]) - self.origin,
          primaryregiony=(primaryregionsy[gc][gy-1], primaryregionsy[gc][gy]) - self.origin,
          readingfromfile=False,
        )
      )
    return result

  @property
  def fields(self):
    return self.__fields()

  def writetable(self, *filenames, rtol=1e-3, atol=1e-5, check=False, logger=dummylogger, **kwargs):
    affinefilename, fieldsfilename, fieldoverlapfilename = filenames

    fields = self.fields
    writetable(fieldsfilename, fields, rowclass=Field, **kwargs)

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

    writetable(fieldoverlapfilename, self.fieldoverlaps, **kwargs)

    if check:
      logger.debug("reading back from the file")
      readback = ReadStitchResult(*filenames, rectangles=self.rectangles, overlaps=self.overlaps, origin=self.origin)
      logger.debug("done reading")
      x1 = self.x()
      T1 = self.T
      x2 = readback.x()
      T2 = readback.T
      logger.debug("comparing nominals")
      units.np.testing.assert_allclose(units.nominal_values(x1), units.nominal_values(x2), atol=atol, rtol=rtol)
      units.np.testing.assert_allclose(units.nominal_values(T1), units.nominal_values(T2), atol=atol, rtol=rtol)
      logger.debug("comparing individual errors")
      units.np.testing.assert_allclose(units.std_devs(x1), units.std_devs(x2), atol=atol, rtol=rtol)
      units.np.testing.assert_allclose(units.std_devs(T1), units.std_devs(T2), atol=atol, rtol=rtol)
      logger.debug("comparing overlap errors")
      for o in self.overlaps:
        units.np.testing.assert_allclose(units.covariance_matrix(self.dx(o)), units.covariance_matrix(readback.dx(o)), atol=atol, rtol=rtol)
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
    ): units.np.testing.assert_allclose(units.std_dev(thing)**2, errorsq)

  def x(self, rectangle_or_id=None):
    if rectangle_or_id is None: return self.__x
    if isinstance(rectangle_or_id, Rectangle): rectangle_or_id = rectangle_or_id.n
    return self.__x[self.rectangledict[rectangle_or_id]]

  def dx(self, overlap):
    return self.x(overlap.p1) - self.x(overlap.p2) - (overlap.x1vec - overlap.x2vec)

  @property
  def fieldoverlaps(self):
    fieldoverlaps = []
    for o in self.overlaps:
      if o.p2 < o.p1: continue
      covariance = np.array(units.covariance_matrix(np.concatenate([self.x(o.p1), self.x(o.p2)])))
      fieldoverlaps.append(
        FieldOverlap(
          overlap=o,
          cov_x1_x2=covariance[0,2],
          cov_x1_y2=covariance[0,3],
          cov_y1_x2=covariance[1,2],
          cov_y1_y2=covariance[1,3],
          readingfromfile=False,
        )
      )
    return fieldoverlaps

class StitchResultOverlapCovariances(StitchResultBase):
  def __init__(self, *, x, T, fieldoverlaps, **kwargs):
    self.__x = x
    self.__T = T
    self.__fieldoverlaps = fieldoverlaps
    super().__init__(**kwargs)

  @property
  def T(self): return self.__T
  @property
  def fieldoverlaps(self): return self.__fieldoverlaps

  def x(self, rectangle_or_id=None):
    if rectangle_or_id is None: return self.__x
    if isinstance(rectangle_or_id, Rectangle): rectangle_or_id = rectangle_or_id.n
    return self.__x[self.rectangledict[rectangle_or_id]]

  def dx(self, overlap):
    x1 = self.x(overlap.p1)
    x2 = self.x(overlap.p2)
    fieldoverlap = self.fieldoverlap(overlap)

    nominals = np.concatenate([units.nominal_values(x1), units.nominal_values(x2)])

    covariance = np.zeros((4, 4), dtype=units.unitdtype)
    covariance[:2,:2] = units.covariance_matrix(x1)
    covariance[2:,2:] = units.covariance_matrix(x2)
    covariance[0,2] = covariance[2,0] = fieldoverlap.cov_x1_x2
    covariance[0,3] = covariance[3,0] = fieldoverlap.cov_x1_y2
    covariance[1,2] = covariance[2,1] = fieldoverlap.cov_y1_x2
    covariance[1,3] = covariance[3,1] = fieldoverlap.cov_y1_y2

    xx1, yy1, xx2, yy2 = units.correlated_distances(distances=nominals, covariance=covariance)
    newx1 = np.array([xx1, yy1])
    newx2 = np.array([xx2, yy2])

    units.np.testing.assert_allclose(units.nominal_values(x1), units.nominal_values(newx1))
    units.np.testing.assert_allclose(units.covariance_matrix(x1), units.covariance_matrix(newx1))
    units.np.testing.assert_allclose(units.nominal_values(x2), units.nominal_values(newx2))
    units.np.testing.assert_allclose(unc.covariance_matrix(x2), units.covariance_matrix(newx2))

    return newx1 - newx2 - (overlap.x1vec - overlap.x2vec)

  @methodtools.lru_cache()
  def __fieldoverlapdict(self):
    return {frozenset((oc.p1, oc.p2)): oc for oc in self.fieldoverlaps}

  def fieldoverlap(self, overlap):
    return self.__fieldoverlapdict()[frozenset((overlap.p1, overlap.p2))]

  def readtable(self, *filenames, adjustoverlaps=True):
    affinefilename, fieldsfilename, fieldoverlapfilename = filenames

    fields = readtable(fieldsfilename, Field, extrakwargs={"pscale": self.pscale})
    affines = readtable(affinefilename, AffineEntry)
    layer, = {_.layer for _ in self.overlaps}
    nclip, = {_.nclip for _ in self.overlaps}
    fieldoverlaps = readtable(fieldoverlapfilename, FieldOverlap, extrakwargs={"pscale": self.pscale, "rectangles": self.rectangles, "layer": layer, "nclip": nclip})

    self.__x = np.array([field.pxvec+self.origin for field in fields])
    self.__fieldoverlaps = fieldoverlaps

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
  def __init__(self, *args, rectangles, overlaps, origin, **kwargs):
    super().__init__(rectangles=rectangles, overlaps=overlaps, x=None, T=None, fieldoverlaps=None, origin=origin)
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
  def __init__(self, *, x, T, problem, pscale, **kwargs):
    super().__init__(
      x=units.distances(pixels=x.value, pscale=pscale),
      T=T.value,
      covariancematrix=np.zeros(x.size+T.size, x.size+T.size),
      **kwargs
    )
    self.problem = problem
    self.xvar = x
    self.Tvar = T

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
