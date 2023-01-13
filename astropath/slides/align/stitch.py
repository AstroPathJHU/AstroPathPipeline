import abc, collections, itertools, methodtools, more_itertools, numpy as np, pathlib, uncertainties as unc
from ...shared.logging import dummylogger
from ...shared.overlap import RectangleOverlapCollection
from ...shared.rectangle import Rectangle, rectangledict
from ...utilities import units
from ...utilities.dataclasses import MetaDataAnnotation
from ...utilities.miscmath import covariance_matrix, floattoint, weightedstd
from ...utilities.optionalimports import cvxpy as cp
from ...utilities.tableio import writetable
from ...utilities.units.dataclasses import DataClassWithPscale, distancefield
from .field import Field, FieldList, FieldOverlap

def stitch(*, usecvxpy=False, **kwargs):
  return (__stitch_cvxpy if usecvxpy else __stitch)(**kwargs)

def __stitch(*, rectangles, overlaps, scalejittererror=1, scaleoverlaperror=1, scaleedges=1, scalecorners=1, fixpoint="origin", origin=np.array([0, 0]), margin, logger=dummylogger, griderroraction="error"):
  """
  stitch the alignment results together

  rectangles: all HPF objects in the slide
  overlaps: all aligned overlaps
  scalejittererror: scale factor for the error on the positions of HPFs \sigma (contributes mainly to the relative positions of islands)
  scaleoverlaperror: scale factor for the error on the overlap alignments \sqrt{\mathbf{cov}}
  scaleedges: weight for the edge overlaps
  scalecorners: weight for the corner overlaps
  fixpoint: the point that is fixed by the fitted affine transformation.  choices: origin (default), center, or an array
  origin: the origin of the coordinate system (used if fixpoint == "origin" and also in defining the aligned coordinate system pxvec)
  logger: the alignment set's logger

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
  nconstraints = 0
  ndof = size
  found_x_overlap = found_y_overlap = False
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

  scaleoverlap = {
    1: scalecorners,
    2: scaleedges,
    3: scalecorners,
    4: scaleedges,
    6: scaleedges,
    7: scalecorners,
    8: scaleedges,
    9: scalecorners,
  }

  for o in overlaps:
    nconstraints += 2
    if o.x1 != o.x2: found_x_overlap = True
    if o.y1 != o.y2: found_y_overlap = True

    #get the indices of the coordinates of the two overlaps to index into A and b
    ix = 2*rd[o.p1]
    iy = 2*rd[o.p1]+1
    jx = 2*rd[o.p2]
    jy = 2*rd[o.p2]+1
    assert ix >= 0, ix
    assert iy < 2*len(rectangles), iy
    assert jx >= 0, jx
    assert jy < 2*len(rectangles), jy

    #contribution of the overlap to A
    ii = np.ix_((ix,iy), (ix,iy))
    ij = np.ix_((ix,iy), (jx,jy))
    ji = np.ix_((jx,jy), (ix,iy))
    jj = np.ix_((jx,jy), (jx,jy))
    inversecovariance = units.np.linalg.inv(o.result.covariance) / scaleoverlaperror**2

    A[ii] += inversecovariance * scaleoverlap[o.tag]
    A[ij] -= inversecovariance * scaleoverlap[o.tag]
    A[ji] -= inversecovariance * scaleoverlap[o.tag]
    A[jj] += inversecovariance * scaleoverlap[o.tag]

    #contribution of the overlap to b
    i = np.ix_((ix, iy))
    j = np.ix_((jx, jy))

    constpiece = (-units.nominal_values(o.result.dxvec) - o.x1vec + o.x2vec)

    b[i] += 2 * inversecovariance @ constpiece * scaleoverlap[o.tag]
    b[j] -= 2 * inversecovariance @ constpiece * scaleoverlap[o.tag]

    #contribution of the overlap to c
    c += constpiece @ inversecovariance @ constpiece * scaleoverlap[o.tag]

  dxs, dys = zip(*(o.result.dxvec for o in overlaps))

  sigmax = weightedstd(dxs, subtractaverage=False) * scalejittererror
  sigmay = weightedstd(dys, subtractaverage=False) * scalejittererror

  if fixpoint == "origin":
    x0vec = origin
  elif fixpoint == "center":
    x0vec = np.mean([r.xvec for r in rectangles], axis=0)
  else:
    x0vec = fixpoint

  x0, y0 = x0vec

  for r in rectangles:
    nconstraints += 2

    ix = 2*rd[r.n]
    iy = 2*rd[r.n]+1

    x = r.x
    y = r.y

    #contribution of the HPF to A
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

    #contribution of the HPF to b
    b[ix] -= 2 * x0 / sigmax**2
    b[iy] -= 2 * y0 / sigmay**2

    b[Txx] += 2 * x0 * (x - x0) / sigmax**2
    b[Txy] += 2 * x0 * (y - y0) / sigmax**2
    b[Tyx] += 2 * y0 * (x - x0) / sigmay**2
    b[Tyy] += 2 * y0 * (y - y0) / sigmay**2

    #contribution of the HPF to c
    c += x0**2 / sigmax**2
    c += y0**2 / sigmay**2

  logger.debug("assembled A b c")

  #if any parameters are fixed, remove the dependence of A and b on those
  #parameters.  The dependence is added to b and c in such a way that, when
  #those parameters are set to the values they're fixed to, the total log
  #likelihood is unchanged.
  fixedindices = np.zeros_like(b, dtype=bool)
  fixedmus = np.zeros_like(b, dtype=float)
  fixedsigmas = np.zeros_like(fixedmus)
  fixedmus[Txx] = fixedmus[Tyy] = 1
  fixedmus[Txy] = fixedmus[Tyx] = 0
  fixedsigmas[Txx] = fixedsigmas[Txy] = fixedsigmas[Tyx] = fixedsigmas[Tyy] = .001
  if not found_x_overlap:
    logger.warning("All HPFs with successful overlaps are in the same row --> fixing x components of the affine matrix")
    fixedindices[Txx] = fixedindices[Tyx] = True
  if not found_y_overlap:
    logger.warning("All HPFs with successful overlaps are in the same column --> fixing y components of the affine matrix")
    fixedindices[Txy] = fixedindices[Tyy] = True
  if nconstraints < ndof - np.count_nonzero(fixedindices):
    logger.warning("Not enough constraints --> fixing the off diagonal elements of the affine matrix to 0")
    fixedindices[Txy] = fixedindices[Tyx] = True
  if nconstraints < ndof - np.count_nonzero(fixedindices):
    logger.warning("Still not enough constraints --> fixing the affine matrix to 1")
    fixedindices[Txx] = fixedindices[Tyy] = True
  if nconstraints < ndof - np.count_nonzero(fixedindices):
    assert False #this should not be able to happen because ndof is now 2*nrectangles and each rectangle is constrained to the affine grid

  floatedindices = ~fixedindices

  floatedindices = np.arange(len(b))[floatedindices]
  fixedindices = np.arange(len(b))[fixedindices]
  fixedmus = fixedmus[fixedindices]
  fixedsigmas = fixedsigmas[fixedindices]

  floatfix = np.ix_(floatedindices, fixedindices)
  fixfloat = np.ix_(fixedindices, floatedindices)
  fixfix = np.ix_(fixedindices, fixedindices)

  #A entries that correspond to 2 fixed parameters: goes into c
  c += fixedmus @ A[fixfix] @ fixedmus
  A[fixfix] = 0

  #A entries that correspond to a fixed parameter and a floated parameter
  b[floatedindices] += A[floatfix] @ fixedmus + fixedmus @ A[fixfloat]
  A[floatfix] = A[fixfloat] = 0

  #b entries that correspond to a fixed parameter
  c += b[fixedindices] @ fixedmus
  b[fixedindices] = 0

  #add the constraints into A, b, c
  for idx, mu, sigma in more_itertools.zip_equal(fixedindices, fixedmus, fixedsigmas):
    assert A[idx,idx] == b[idx] == 0
    A[idx, idx] += 1/sigma**2
    b[idx] -= 2*mu/sigma**2
    c += (mu/sigma)**2

  result = units.np.linalg.solve(2*A, -b)

  logger.debug("solved quadratic equation")

  delta2nllfor1sigma = 1

  covariancematrix = units.np.linalg.inv(A) * delta2nllfor1sigma
  logger.debug("got covariance matrix")
  result = np.array(units.correlated_distances(distances=result, covariance=covariancematrix))

  x = result[:-4].reshape(len(rectangles), 2)
  T = result[-4:].reshape(2, 2)

  logger.debug("done")

  return StitchResult(x=x, T=T, A=A, b=b, c=c, rectangles=rectangles, overlaps=alloverlaps, covariancematrix=covariancematrix, origin=origin, margin=margin, logger=logger, griderroraction=griderroraction)

def __stitch_cvxpy(*, overlaps, rectangles, fixpoint="origin", origin=np.array([0, 0]), margin, logger=dummylogger, griderroraction="error"):
  """
  Run the stitching using cvxpy as a cross check.
  Arguments are the same as __stitch, with some limitations.
  Most notably, the error on the results will not be available.

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
    #contribution to the quadratic from the overlap
    x1 = rectanglex[o.p1]
    x2 = rectanglex[o.p2]
    twonll += cp.quad_form(
      x1 - x2 + (-units.nominal_values(o.result.dxvec) - o.x1vec + o.x2vec) / o.onepixel,
      (units.np.linalg.inv(o.result.covariance)) * o.onepixel**2
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
    #contribution to the quadratic from the HPF
    twonll += cp.norm(
      (
        (rectanglex[r.n] - x0vec / r.onepixel)
        - T @ ((r.xvec - x0vec) / r.onepixel)
      ) / (sigma / r.onepixel)
    )

  minimize = cp.Minimize(twonll)
  prob = cp.Problem(minimize)
  prob.solve()

  return StitchResultCvxpy(x=x, T=T, problem=prob, rectangles=rectangles, overlaps=alloverlaps, pscale=pscale, origin=origin, margin=margin, logger=logger, griderroraction=griderroraction)

class StitchResultBase(RectangleOverlapCollection, units.ThingWithPscale):
  """
  Base class for a stitching result.

  rectangles: the HPFs
  overlaps: the overlaps
  origin: the origin for defining the pxvec coordinate system
  logger: the AlignSample's logger
  griderror: what to do if the HPFs don't align to the grid
    "error": raise an error
    "show": display an image of how the HPFs look (meant for jupyter)
    pathlib.Path object: save an image of how the HPFs look to a file
  """
  def __init__(self, *, rectangles, overlaps, origin, margin, logger=dummylogger, griderroraction="error"):
    self.__rectangles = rectangles
    self.__overlaps = overlaps
    self.__origin = origin
    self.__margin = margin
    self.__logger = logger

    self.__griderroraction = griderroraction
    self.__griderror = None
    if not isinstance(griderroraction, pathlib.Path) and griderroraction != "error" and griderroraction != "show":
      raise ValueError("griderroraction should be 'error', 'show', or a pathlib.Path")

  @methodtools.lru_cache()
  @property
  def gridatol(self):
    shape = np.min([_.shape for _ in self.rectangles], axis=0)
    return shape / 10

  @methodtools.lru_cache()
  @property
  def pscale(self):
    pscale, = {_.pscale for _ in itertools.chain(self.rectangles, self.overlaps)}
    return pscale

  @property
  def overlaps(self): return self.__overlaps
  @property
  def rectangles(self): return self.__rectangles
  @property
  def origin(self): return self.__origin
  @property
  def margin(self): return self.__margin

  @abc.abstractmethod
  def x(self, rectangle_or_id=None):
    """
    The stitched position of the HPF
    """
  @abc.abstractmethod
  def dx(self, overlap):
    """
    The stitched relative position of the two HPFs of the overlap.
    Not necessarily the same as x(i1) - x(i2).  The nominal value will be
    but the error may not if the full covariance matrix is not stored.
    """
  @property
  @abc.abstractmethod
  def T(self):
    """
    The affine matrix
    """
  @property
  @abc.abstractmethod
  def fieldoverlaps(self):
    """
    the field overlaps with the covariance entries
    """

  def applytooverlaps(self):
    """
    Give all the overlaps a stitchresult vector
    """
    for o in self.overlaps:
      o.stitchresult = self.dx(o)

  @methodtools.lru_cache()
  @property
  def __fields_and_shift(self):
    """
    Create the field objects from the rectangles and stitch result
    """
    result = FieldList()
    gridislands = list(self.islands(useexitstatus=False, gridatol=self.gridatol))
    alignedislands = list(self.islands(useexitstatus=True, gridatol=None))
    onepixel = self.onepixel
    gxdict = collections.defaultdict(dict)
    gydict = collections.defaultdict(dict)
    primaryregionsx = {}
    primaryregionsy = {}
    isorphan = {}
    adjustcx = collections.defaultdict(dict)
    adjustcy = collections.defaultdict(dict)

    shape = {tuple(r.shape) for r in self.rectangles}
    if len(shape) > 1:
      raise ValueError("Some rectangles have different shapes")
    shape = shape.pop()

    for gc, island in enumerate(gridislands, start=1):
      rectangles = [self.rectangles[self.rectangledict[n]] for n in island]
      isorphan[gc] = len(island) == 1 and island in alignedislands

      for i, (primaryregions, gdict, adjustc) in enumerate(zip((primaryregionsx, primaryregionsy), (gxdict, gydict), (adjustcx[gc], adjustcy[gc]))):
        #find the gx and gy that correspond to each nominal x and y
        average = []

        gridatol = self.gridatol[i]
        allcs = sorted({r.xvec[i] for r in rectangles})
        adjustc.update({c: c for c in allcs})
        for c1, c2 in more_itertools.pairwise(allcs):
          if c2-c1 <= gridatol:
            adjustc[c2] = adjustc[c1]
            if c2 - adjustc[c2] > gridatol:
              raise ValueError(f"HPFs are offset from the grid by more than the tolerance {c2} {adjustc[c2]}, please check")
        cs = sorted(set(adjustc.values()))

        for g, c in enumerate(cs, start=1):
          gdict[gc][c] = g
          theserectangles = [r for r in rectangles if adjustc[r.xvec[i]] == c]
          average.append(np.mean(units.nominal_values([self.x(r)[i] for r in theserectangles])))

        #find mx1, my1, mx2, my2
        #the middle ones come from the average positions of the HPFs on either side
        primaryregions[gc] = [(x1+shape[i] + x2)/2 for x1, x2 in more_itertools.pairwise(average)]

        if len(primaryregions[gc]) >= 2:
          #the outer ones come from fitting a line to the middle ones
          #however sometimes there's a jump in the microscope at some point
          #and we want to only use the ones on the correct side of the jump
          diffs = np.diff(primaryregions[gc])
          averagediff = np.mean(diffs)
          stddiff = np.std(diffs)
          m = np.mean(diffs[abs(diffs-averagediff)<=stddiff])

          bs_left = []
          for i, primaryregionboundary in enumerate(primaryregions[gc], start=1):
            b = primaryregionboundary - m*i
            if len(bs_left) >= 2:
              b_residual = (b - np.mean(bs_left))
              if abs(b_residual / (np.std(bs_left) / np.sqrt(len(bs_left)))) > 100:
                break
            bs_left.append(b)
          b_left = np.mean(bs_left)

          bs_right = []
          for i, primaryregionboundary in reversed(list(enumerate(primaryregions[gc], start=1))):
            b = primaryregionboundary - m*i
            if len(bs_right) >= 2:
              b_residual = (b - np.mean(bs_right))
              if abs(b_residual / (np.std(bs_right) / np.sqrt(len(bs_right)))) > 100:
                break
            bs_right.append(b)
          b_right = np.mean(bs_right)

          primaryregions[gc].insert(0, m*0+b_left)
          primaryregions[gc].append(m*len(average)+b_right)
        else:
          #can't fit a line because there are only at most 2 rows/columns, so do an approximation
          if len(primaryregions[gc]) == 1:
            primaryregions[gc].insert(0, primaryregions[gc][0] - self.hpfoffset[i])
            primaryregions[gc].append(primaryregions[gc][1] + self.hpfoffset[i])
          else: #len(primaryregions[gc]) == 0
            primaryregions[gc].append(average[0] + (shape[i] - self.hpfoffset[i]) / 2)
            primaryregions[gc].append(average[0] + (shape[i] + self.hpfoffset[i]) / 2)

    mx1 = {}
    mx2 = {}
    my1 = {}
    my2 = {}

    #set gx, gy, mx1, my1, mx2, my2 for the HPFs
    for i, island in enumerate(gridislands, start=1):
      for rid in island:
        r = self.rectangles[self.rectangledict[rid]]

        gx = gxdict[i][adjustcx[i][r.x]]
        gy = gydict[i][adjustcy[i][r.y]]

        mx1[rid] = primaryregionsx[i][gx-1]
        mx2[rid] = primaryregionsx[i][gx]
        my1[rid] = primaryregionsy[i][gy-1]
        my2[rid] = primaryregionsy[i][gy]

        pxvec = units.nominal_values(self.x(r))

        neighbors = self.neighbors(r, useexitstatus=False)
        if not neighbors.keys() & {1, 4, 7}: mx1[rid] = pxvec[0]
        if not neighbors.keys() & {1, 2, 3}: my1[rid] = pxvec[1]
        if not neighbors.keys() & {3, 6, 9}: mx2[rid] = pxvec[0]+r.w
        if not neighbors.keys() & {7, 8, 9}: my2[rid] = pxvec[1]+r.h

        if mx1[rid] < pxvec[0]:
          msg = f"{rid}: px = {pxvec[0]}, mx1 = {mx1[rid]}"
          if gx == 1:
            self.__logger.warning(f"{msg}, adjusting mx1")
            mx1[rid] = pxvec[0]
          else:
            if self.__griderroraction == "error":
              raise ValueError(msg)
            else:
              self.__logger.warning(msg)
            self.__griderror = msg
        if mx2[rid] > pxvec[0]+r.w:
          msg = f"{rid}: px+w = {pxvec[0]+r.w}, mx2 = {mx2[rid]}"
          if gx == max(gxdict[i].values()):
            self.__logger.warning(f"{msg}, adjusting mx2")
            mx2[rid] = pxvec[0]+r.w
          else:
            if self.__griderroraction == "error":
              raise ValueError(msg)
            else:
              self.__logger.warning(msg)
            self.__griderror = msg

        if my1[rid] < pxvec[1]:
          msg = f"{rid}: py = {pxvec[1]}, my1 = {my1[rid]}"
          if gy == 1:
            self.__logger.warning(f"{msg}, adjusting my1")
            my1[rid] = pxvec[1]
          else:
            if self.__griderroraction == "error":
              raise ValueError(msg)
            else:
              self.__logger.warning(msg)
            self.__griderror = msg

        if my2[rid] > pxvec[1]+r.h:
          msg = f"{rid}: py+h = {pxvec[1]+r.h}, my2 = {my2[rid]}"
          if gy == max(gydict[i].values()):
            self.__logger.warning(f"{msg}, adjusting my2")
            my2[rid] = pxvec[1]+r.h
          else:
            if self.__griderroraction == "error":
              raise ValueError(msg)
            else:
              self.__logger.warning(msg)
            self.__griderror = msg

    #see if the primary regions of any HPFs in different islands overlap
    for (i1, island1), (i2, island2) in itertools.combinations(enumerate(gridislands, start=1), r=2):
      if isorphan[i1] or isorphan[i2]: continue #orphans are excluded

      #first see if the islands overlap
      x11 = min(primaryregionsx[i1])
      x21 = max(primaryregionsx[i1])
      x12 = min(primaryregionsx[i2])
      x22 = max(primaryregionsx[i2])

      y11 = min(primaryregionsy[i1])
      y21 = max(primaryregionsy[i1])
      y12 = min(primaryregionsy[i2])
      y22 = max(primaryregionsy[i2])

      #if a box around the islands overlaps in both x and y
      if (
        max(x21, x22) - min(x11, x12) + 1e-5*x11 < (x21 - x11) + (x22 - x12)
        and max(y21, y22) - min(y11, y12) + 1e-5*x11 < (y21 - y11) + (y22 - y12)
      ):
        self.__logger.info(f"Primary regions for islands {i1} and {i2} overlap in both x and y, seeing if any field primary regions overlap")

        xoverlapstoadjust = collections.defaultdict(list)
        yoverlapstoadjust = collections.defaultdict(list)
        cornerstoadjust = collections.defaultdict(list)

        for rid1, rid2 in itertools.product(island1, island2):
          xx11 = mx1[rid1]
          xx21 = mx2[rid1]
          xx12 = mx1[rid2]
          xx22 = mx2[rid2]

          yy11 = my1[rid1]
          yy21 = my2[rid1]
          yy12 = my1[rid2]
          yy22 = my2[rid2]

          if (
            max(xx21, xx22) - min(xx11, xx12) + 1e-5*x11 < (xx21 - xx11) + (xx22 - xx12)
            and max(yy21, yy22) - min(yy11, yy12) + 1e-5*x11 < (yy21 - yy11) + (yy22 - yy12)
          ):
            self.__logger.warningglobal(f"Primary regions for fields {rid1} and {rid2} overlap, adjusting them")

            threshold = 1 / 10
            if len(island1) <= 2 or len(island2) <= 2:
              threshold = 1 / 4
            possiblexs = possibleys = xs = ys = None
            ridax = ridbx = riday = ridby = None

            if abs(xx21 - xx12) <= abs(xx11 - xx22):
              possiblexs = xx12, xx21
              ridax, ridbx = rid2, rid1
            else:
              possiblexs = xx11, xx22
              ridax, ridbx = rid1, rid2
            if abs(yy21 - yy12) <= abs(yy11 - yy22):
              possibleys = yy12, yy21
              riday, ridby = rid2, rid1
            else:
              possibleys = yy11, yy22
              riday, ridby = rid1, rid2

            fractionaloffset = [abs(possiblexs[0] - possiblexs[1]), abs(possibleys[0] - possibleys[1])] / self.hpfoffset

            if fractionaloffset[0] <= threshold:
              xs = possiblexs
            if fractionaloffset[1] <= threshold:
              ys = possibleys

            if xs is ys is None:
              self.__logger.warningglobal(f"Primary regions for fields {rid1} and {rid2} have a very large overlap, please check the output primary regions")
              self.__logger.warning(f"field {rid1}: mx = ({xx11}, {xx21}), my = ({yy11}, {yy21})")
              self.__logger.warning(f"field {rid2}: mx = ({xx12}, {xx22}), my = ({yy12}, {yy22})")
              if fractionaloffset[0] > 1.5*fractionaloffset[1]:
                ys = possibleys
              elif fractionaloffset[1] > 1.5*fractionaloffset[0]:
                xs = possiblexs
              else:
                xs = possiblexs
                ys = possibleys

            if xs is not None and ys is not None:
              cornerstoadjust[xs, ys].append((ridax, ridbx, riday, ridby))
            elif xs is not None:
              xoverlapstoadjust[xs].append((ridax, ridbx))
            elif ys is not None:
              yoverlapstoadjust[ys].append((riday, ridby))

        cornerxscounter = collections.Counter(xs for xs, ys in cornerstoadjust)
        corneryscounter = collections.Counter(ys for xs, ys in cornerstoadjust)
        xswith2corners = [xs for xs, count in cornerxscounter.items() if count >= 2]
        yswith2corners = [ys for ys, count in corneryscounter.items() if count >= 2]

        for (xs, ys), rids in cornerstoadjust.items():
          if xs in xswith2corners:
            xoverlapstoadjust[xs] += [(ridax, ridbx) for ridax, ridbx, riday, ridby in rids]
          if ys in yswith2corners:
            yoverlapstoadjust[ys] += [(riday, ridby) for ridax, ridbx, riday, ridby in rids]
        for (xs, ys), rids in cornerstoadjust.items():
          if xs in xoverlapstoadjust:
            xoverlapstoadjust[xs] += [(ridax, ridbx) for ridax, ridbx, riday, ridby in rids]
          if ys in yoverlapstoadjust:
            yoverlapstoadjust[ys] += [(riday, ridby) for ridax, ridbx, riday, ridby in rids]

        for ((oldmx1, oldmx2), (oldmy1, oldmy2)), rids in cornerstoadjust.items():
          if (oldmx1, oldmx2) in xoverlapstoadjust or (oldmy1, oldmy2) in yoverlapstoadjust:
            pass
          elif oldmx1 - oldmx2 < oldmy1 - oldmy2:
            xoverlapstoadjust[oldmx1, oldmx2] += [(ridax, ridbx) for ridax, ridbx, riday, ridby in rids]
          else:
            yoverlapstoadjust[oldmy1, oldmy2] += [(riday, ridby) for ridax, ridbx, riday, ridby in rids]

        for (oldmx1, oldmx2), rids in xoverlapstoadjust.items():
          for rid1, rid2 in rids:
            newmx = (oldmx1 + oldmx2)/2
            if not ((mx1[rid1] == oldmx1 or mx1[rid1] == newmx) and (mx2[rid2] == oldmx2 or mx2[rid2] == newmx)):
              msg = f"????? {mx1[rid1]} {oldmx1} {mx2[rid2]} {oldmx2} {newmx}"
              if self.__griderroraction == "error":
                raise ValueError(msg)
              else:
                self.__logger.warning(msg)
              self.__griderror = msg
            mx1[rid1] = mx2[rid2] = newmx
        for (oldmy1, oldmy2), rids in yoverlapstoadjust.items():
          for rid1, rid2 in rids:
            newmy = (oldmy1 + oldmy2)/2
            if not ((my1[rid1] == oldmy1 or my1[rid1] == newmy) and (my2[rid2] == oldmy2 or my2[rid2] == newmy)):
              msg = f"????? {my1[rid1]} {oldmy1} {my2[rid2]} {oldmy2} {newmy}"
              if self.__griderroraction == "error":
                raise ValueError(msg)
              else:
                self.__logger.warning(msg)
              self.__griderror = msg
            my1[rid1] = my2[rid2] = newmy

        for rid1, rid2 in itertools.product(island1, island2):
          xx11 = mx1[rid1]
          xx21 = mx2[rid1]
          xx12 = mx1[rid2]
          xx22 = mx2[rid2]

          yy11 = my1[rid1]
          yy21 = my2[rid1]
          yy12 = my1[rid2]
          yy22 = my2[rid2]

          if (
            max(xx21, xx22) - min(xx11, xx12) + 1e-5*x11 < (xx21 - xx11) + (xx22 - xx12)
            and max(yy21, yy22) - min(yy11, yy12) + 1e-5*x11 < (yy21 - yy11) + (yy22 - yy12)
          ):
            msg = f"Primary regions for fields {rid1} and {rid2} still overlap"
            if self.__griderroraction == "error":
              raise ValueError(msg)
            else:
              self.__logger.warning(msg)
            self.__griderror = msg

    minpxvec = [np.inf * onepixel, np.inf * onepixel]

    for rectangle in self.rectangles:
      for gc, island in enumerate(gridislands, start=1):
        if rectangle.n in island:
          break
      else:
        assert False
      gx = gxdict[gc][adjustcx[gc][rectangle.x]]
      gy = gydict[gc][adjustcy[gc][rectangle.y]]
      pxvec = self.x(rectangle) - self.origin
      minpxvec = np.min([minpxvec, units.nominal_values(pxvec)], axis=0)
      result.append(
        Field(
          rectangle=rectangle,
          ixvec=floattoint(np.round((rectangle.xvec / onepixel).astype(float))) * onepixel,
          gc=0 if isorphan[gc] else gc,
          pxvec=pxvec,
          gxvec=(gx, gy),
          primaryregionx=np.array([mx1[rectangle.n], mx2[rectangle.n]]) - self.origin[0],
          primaryregiony=np.array([my1[rectangle.n], my2[rectangle.n]]) - self.origin[1],
          readingfromfile=False,
        )
      )

    #if there are any HPFs that are in the wrong quadrant (negative px or py), adjust the whole slide
    minx, miny = np.floor((minpxvec - self.margin)/(100*onepixel))*100*onepixel
    if minx > 0: minx = 0
    if miny > 0: miny = 0
    if minx or miny:
      self.__logger.warningglobal(f"Some HPFs have (x, y) < (xposition, yposition) + margin, shifting the whole slide by ({-minx/self.onepixel}, {-miny/self.onepixel})")
      x = self.x()
      x -= ((minx, miny))
      for f in result:
        f.pxvec -= (minx, miny)
        f.primaryregionx -= minx
        f.primaryregiony -= miny
    shift = -minx, -miny

    if self.__griderror is not None:
      if self.__griderroraction == "show":
        result.showalignedrectanglelayout()
      else:
        result.showalignedrectanglelayout(saveas=self.__griderroraction)
      raise ValueError("Error in aligning HPFs to a grid, see plot and warnings above")

    return result, shift

  @property
  def fields(self):
    return self.__fields_and_shift[0]
  @property
  def globalshift(self):
    return self.__fields_and_shift[1]
  @property
  def fieldboundaries(self):
    return [field.boundary for field in self.fields]

  def writetable(self, *filenames, rtol=1e-3, atol=1e-5, check=False, **kwargs):
    """
    Write the affine, fields, fieldoverlaps, and fieldGeometry csvs

    check: cross check that reading the csvs back gives the same result
           this is nontrivial because we lose part of the covariance matrix
           (cross talk between non-adjacent HPFs and between HPFs and the affine
           matrix) in this procedure
    rtol, atol: tolerance for the cross check
    """

    affinefilename, fieldsfilename, fieldoverlapfilename, fieldgeometryfilename = filenames

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
            description="a"+rowcoordinate+columncoordinate,
            pscale=self.pscale,
          )
        )
    for coordinate, shift in zip("xy", self.globalshift):
      n += 1
      affine.append(
        AffineNominalEntry(
          n=n,
          matrixentry=shift,
          description="shift"+coordinate,
          pscale=self.pscale,
        )
      )

    for entry1, entry2 in itertools.combinations_with_replacement(affine[:], 2):
      n += 1
      entry = AffineCovarianceEntry(n=n, entry1=entry1, entry2=entry2)
      if entry: affine.append(entry)
    writetable(affinefilename, affine, rowclass=AffineEntry, **kwargs)

    writetable(fieldoverlapfilename, self.fieldoverlaps, **kwargs)
    writetable(fieldgeometryfilename, self.fieldboundaries, **kwargs)

    if check:
      self.__logger.debug("reading back from the file")
      readback = ReadStitchResult(*filenames, rectangles=self.rectangles, overlaps=self.overlaps, origin=self.origin, margin=self.margin, logger=self.__logger)
      self.__logger.debug("done reading")
      x1 = self.x()
      T1 = self.T
      x2 = readback.x()
      T2 = readback.T
      self.__logger.debug("comparing nominals")
      units.np.testing.assert_allclose(units.nominal_values(x1), units.nominal_values(x2), atol=atol, rtol=rtol)
      units.np.testing.assert_allclose(units.nominal_values(T1), units.nominal_values(T2), atol=atol, rtol=rtol)
      self.__logger.debug("comparing individual errors")
      units.np.testing.assert_allclose(units.std_devs(x1), units.std_devs(x2), atol=atol, rtol=rtol)
      units.np.testing.assert_allclose(units.std_devs(T1), units.std_devs(T2), atol=atol, rtol=rtol)
      self.__logger.debug("comparing overlap errors")
      for o in self.overlaps:
        units.np.testing.assert_allclose(units.covariance_matrix(self.dx(o)), units.covariance_matrix(readback.dx(o)), atol=atol, rtol=rtol)
      self.__logger.debug("done")

class StitchResultFullCovariance(StitchResultBase):
  """
  Base class for a stitch result that has the full covariance matrix
  """
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
  """
  Base class for a stitch result that only keeps the covariance entries
  for adjacent HPFs
  """
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
    units.np.testing.assert_allclose(units.covariance_matrix(x2), units.covariance_matrix(newx2))

    return newx1 - newx2 - (overlap.x1vec - overlap.x2vec)

  @methodtools.lru_cache()
  def __fieldoverlapdict(self):
    return {frozenset((oc.p1, oc.p2)): oc for oc in self.fieldoverlaps}

  def fieldoverlap(self, overlap):
    return self.__fieldoverlapdict()[frozenset((overlap.p1, overlap.p2))]

  def readtables(self, *filenames, adjustoverlaps=True):
    affinefilename, fieldsfilename, fieldoverlapfilename, fieldgeometryfilename = filenames

    layer, = {_.alignmentlayer for _ in self.rectangles}
    fields = self.readtable(fieldsfilename, Field)
    affines = self.readtable(affinefilename, AffineEntry)
    nclip, = {_.nclip for _ in self.overlaps}
    fieldoverlaps = self.readtable(fieldoverlapfilename, FieldOverlap, extrakwargs={"rectangles": self.rectangles, "nclip": nclip})

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

    shiftx = dct.get("shiftx", 0)
    shifty = dct.get("shifty", 0)
    assert not any("cov_" in _ and "shift" in _ for _ in dct)

    self.__T = np.array(unc.correlated_values(Tnominal, Tcovariance)).reshape((2, 2))
    self.__initialshift = np.array((shiftx, shifty))

  @property
  def globalshift(self):
    return self.__initialshift + super().globalshift

class ReadStitchResult(StitchResultOverlapCovariances):
  """
  Stitch result that reads from csvs
  """
  def __init__(self, *args, rectangles, overlaps, origin, margin, logger=dummylogger, **kwargs):
    super().__init__(rectangles=rectangles, overlaps=overlaps, x=None, T=None, fieldoverlaps=None, origin=origin, margin=margin, logger=logger)
    self.readtables(*args, **kwargs)

class CalculatedStitchResult(StitchResultFullCovariance):
  """
  Stitch result that is calculated using the stitching algorithm
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sanitycheck()

class StitchResult(CalculatedStitchResult):
  """
  Stitch result using the standalone stitching algorithm with A, b, and c
  """
  def __init__(self, *, A, b, c, **kwargs):
    super().__init__(**kwargs)
    self.A = A
    self.b = b
    self.c = c

class StitchResultCvxpy(CalculatedStitchResult):
  """
  Stitch result using cvxpy
  """
  def __init__(self, *, x, T, problem, pscale, **kwargs):
    onepixel = units.onepixel(pscale=pscale)
    super().__init__(
      x=x.value*onepixel,
      T=T.value,
      covariancematrix=np.zeros(x.size+T.size, x.size+T.size),
      **kwargs
    )
    self.problem = problem
    self.xvar = x
    self.Tvar = T

class AffineEntry(DataClassWithPscale):
  """
  Entry in the affine matrix or in its covariance matrix

  n: index of the entry
  value: value of the entry
  description: description of the entry
  """
  n: int
  value: units.Distance = distancefield(
    pixelsormicrons="pixels",
    power=lambda self: self.description.count("shift"),
  )
  description: str

class AffineNominalEntry(AffineEntry):
  """
  Entry in the affine matrix
  """
  matrixentry: unc.ufloat = MetaDataAnnotation(includeintable=False)

  @classmethod
  def transforminitargs(cls, *, matrixentry, **kwargs):
    if units.std_dev(matrixentry) == 0:
      matrixentry = units.ufloat(matrixentry, 0)
    return super().transforminitargs(value=matrixentry.n, matrixentry=matrixentry, **kwargs)

class AffineCovarianceEntry(AffineEntry):
  """
  Entry in the covariance matrix of the affine matrix
  """
  entry1: unc.ufloat = MetaDataAnnotation(includeintable=False)
  entry2: unc.ufloat = MetaDataAnnotation(includeintable=False)

  @classmethod
  def transforminitargs(cls, *, entry1, entry2, **kwargs):
    if entry1 is entry2:
      value = entry1.matrixentry.s**2
    elif entry1.matrixentry.s == 0 or entry2.matrixentry.s == 0:
      value = 0
    else:
      value = covariance_matrix([entry1.matrixentry, entry2.matrixentry])[0,1]

    morekwargs = {}
    pscales = {entry1.pscale, entry2.pscale}
    pscales.discard(None)
    if pscales and "pscale" not in kwargs:
      morekwargs["pscale"], = pscales

    return super().transforminitargs(value=value, description = "cov_"+entry1.description+"_"+entry2.description, entry1=entry1, entry2=entry2, **kwargs, **morekwargs)

  def __bool__(self):
    return bool(self.value)
