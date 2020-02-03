import cv2, functools, logging, more_itertools, numpy as np, scipy.interpolate, scipy.optimize, skimage.filters

logger = logging.getLogger("align")

def computeshift(images):
  widesearcher = ShiftSearcher(images, smoothsigma=4.0)
  finesearcher = ShiftSearcher(images, smoothsigma=1.5)

  #first a 17-wide smooth grid with big steps
  result = widesearcher.search(
    nx=5, xmin=-8, xmax=8,
    ny=5, ymin=-8, ymax=8,
    x0=0, y0=0,
  )

  done = False

  xmin = ymin = float("inf")
  xmax = ymax = -float("inf")

  i = 5

  while True:
    prevresult = result
    x0 = int(np.round(-prevresult.dx))
    y0 = int(np.round(-prevresult.dy))

    oldxmin, oldxmax, oldymin, oldymax = xmin, xmax, ymin, ymax

    xmin = int(min(xmin, -prevresult.dx - i))
    ymin = int(min(ymin, -prevresult.dy - i))
    xmax = int(max(xmax, -prevresult.dx + i))
    ymax = int(max(ymax, -prevresult.dy + i))

    #if (oldxmin, oldxmax, oldymin, oldymax) == (xmin, xmax, ymin, ymax):
    #  break

    result = finesearcher.search(
      nx=xmax-xmin+1, xmin=xmin, xmax=xmax,
      ny=ymax-ymin+1, ymin=ymin, ymax=ymax,
      x0=x0, y0=y0,
    )
    result.prevresult = prevresult

    logger.debug("%g %g %g %g %g %g", result.dx, result.dy, result.dv, result.R_error_stat, result.R_error_syst, result.F_error)

    if xmin+1 <= -result.dx <= xmax-1 and ymin+1 <= -result.dy <= ymax-1:
      break

  return result

class ShiftSearcher:
  def __init__(self, images, smoothsigma=None):
    #images: dimensions of intensity, index dimensions of length
    #smoothsigma: dimensions of length
    self.images = images

    self.a, self.b = images

    #smooth the images
    if smoothsigma is not None:
      self.a = skimage.filters.gaussian(self.a, sigma=smoothsigma, mode = 'nearest')
      self.b = skimage.filters.gaussian(self.b, sigma=smoothsigma, mode = 'nearest')

    #rescale the intensity
    mse1 = mse(self.a)    #dimensions of intensity**2
    mse2 = mse(self.b)    #dimensions of intensity**2
    s = (mse1*mse2)**0.25 #dimensions of intensity
    self.a *= s/np.sqrt(mse1)  #factor is dimensionless, a still has dimensions of intensity
    self.b *= s/np.sqrt(mse2)  #factor is dimensionless, b still has dimensions of intensity

    self.shifted_arrays = functools.lru_cache()(self.__shifted_arrays)
    self.shifted_array_difference = functools.lru_cache()(self.__shifted_array_difference)
    self.shifted_array_average = functools.lru_cache()(self.__shifted_array_average)
    self.evalkernel = np.vectorize(
      functools.lru_cache()(self.__evalkernel),
      excluded={"nbins"}
    )


  def search(self, nx, xmin, xmax, ny, ymin, ymax, x0, y0, minimizetolerance=1e-7):
    """
    Take the two images a, b, and find their relative shifts.
    a and b are the two images, smoothsigma is the smoothing length,
    nx and ny are the number of points in x and y,
    (xmin, xmax) and (ymin, ymax) are the grid limits
    x0,y0 is the relative shift between the centers.
    The function fits a bicubic spline to the grids, then does
    a gradient search on the interpolated function for the
    minimum.  Returns a struct containing the final shift,
    and the the grid points for debugging.
    """
    #nx, ny are dimensionless
    #xmin, xmax, ymin, ymax, x0, y0 have dimensions of length
    #minimizetolerance has dimensions of intensity

    #create the grid and do brute force evaluations
    gx = np.linspace(xmin, xmax, nx, dtype=int)   #dimensions of length
    gy = np.linspace(ymin, ymax, ny, dtype=int)   #dimensions of length
    x, y = np.meshgrid(gx,gy)                     #still dimensions of length

    result = scipy.optimize.OptimizeResult()

    v = self.evalkernel(x, y)                     #dimensions of intensity
    result.x0 = x0                                #dimensions of length
    result.y0 = y0                                #dimensions of length

    #fit cubic spline to the cost fn
    spline = result.spline = makespline(x, y, v)  #spline(length, length) = intensity

    #find lowest point for inititalization of the gradient search
    minindices = np.unravel_index(np.argmin(v), v.shape)
    result.xc = xc = float(x[minindices])         #dimensions of length
    result.yc = yc = float(y[minindices])         #dimensions of length

    result.update(scipy.optimize.minimize(
      fun=lambda xy: spline(*xy)[0,0],            #dimensions of intensity
      x0=(xc, yc),                                #dimensions of length
      jac=lambda xy: np.array([spline(*xy, dx=1)[0,0], spline(*xy, dy=1)[0,0]]),  #dimensions of intensity/length
      tol=minimizetolerance,                      #dimensions of intensity
      bounds=((xmin, xmax), (ymin, ymax)),        #dimensions of length
      method="TNC",
    ))

    #calculating error according to https://www.osti.gov/servlets/purl/934781
    #first: R-error from eq. (10)
    Delta_t = 1  #dimensions of length

    #two parts of R-error, which we'll store separately
    #and at the end add in quadrature:
    #  (a) statistical error from random fluctuations in intensity
    #      estimate this from the standard deviation of (a-b)
    #      but only from the central part where (a-b) < (a+b)/2
    #  (b) systematic error from the edges of cells when the alignment
    #      (or warping) isn't perfect.  That's characterized by large (a-b).

    newa, newb = self.shifted_arrays(*result.x)
    dd = self.shifted_array_difference(*result.x)
    average = self.shifted_array_average(*result.x)

    #first find (a)

    """
    \begin{align}
    \mathtt{evalkernel}^2 = K^2 &= \frac{1}{n} \sum_i (a_i - b_i)^2 \\
    (\delta(K^2))^2 &= \frac{1}{n^2} \sum_i 4(a_i-b_i)^2((\delta a_i)^2+(\delta b_i)^2) \\
    \delta(K^2) &= \frac{2}{n}\sqrt{\sum_i (a_i-b_i)^2((\delta a_i)^2+(\delta b_i)^2)} \\
    \delta K &= \frac{\delta(K^2)}{2K}
    \end{align}
    """
    K = self.evalkernel(*result.x)
    spline_for_stat_error_on_pixel = self.evalkernel(*result.x, nbins=np.prod(self.a.shape)//2000)[()]
    delta_Ksquared_stat = 2 / np.prod(self.a.shape) * np.sqrt(
      np.sum(
        dd**2 * (spline_for_stat_error_on_pixel(newa)**2 + spline_for_stat_error_on_pixel(newb)**2)
      )
    )
    sigma_e_stat = delta_Ksquared_stat / (2*K)
    kj = []
    deltav = np.zeros(v.shape)
    for idx in np.ndindex(v.shape):
      deltav[idx] = 1
      deltaspline = makespline(x, y, deltav)           #spline(length, length) = dimensionless
      kj.append(deltaspline(*result.x))        #dimensionless
      deltav[idx] = 0
    R_error_stat = Delta_t * sigma_e_stat * np.linalg.norm(kj)   #dimensions of intensity

    #systematic R error, 0 for now
    delta_Ksquared_syst = 2 / np.prod(self.a.shape) * np.sqrt(
      np.sum(
        dd**2 * np.where(abs(dd) > average, dd**2, 0.)
      )
    )
    sigma_e_syst = delta_Ksquared_syst / (2*K)
    R_error_syst = Delta_t * sigma_e_syst * np.linalg.norm(kj)
    logger.debug("%g %g %g %g", delta_Ksquared_syst, K, sigma_e_syst, R_error_syst)

    """
    #F-error from section V
    Kprimespline = makespline(x, y, v, ((xmin+xmax)/2,), ((ymin+ymax)/2,))
    maximizeerror = scipy.optimize.differential_evolution(
      func=lambda xy: -abs(spline(*xy) - Kprimespline(*xy))[0,0],
      bounds=((xmin, xmax), (ymin, ymax)),
    )
    #the paper has a factor of 0.5 in this formula
    #but that's for a 1D spline
    #see Ferrormontecarlo.py, which reproduces their results
    #for the 1D case and shows that the factor is 1 for a 2D spline
    #when the correlation between x and y is small.
    F_error = -maximizeerror.fun
    """
    #not using F-error because it produces a major overestimate
    F_error = 0

    #https://arxiv.org/pdf/hep-ph/0008191.pdf
    hessian = 0.5 * np.array([
      [spline(*result.x, dx=2, dy=0)[0,0], spline(*result.x, dx=1, dy=1)[0,0]],
      [spline(*result.x, dx=1, dy=1)[0,0], spline(*result.x, dx=0, dy=2)[0,0]],
    ])   #dimensions of intensity / length^2
    covariancematrix = (F_error**2 + R_error_stat**2 + R_error_syst**2 + minimizetolerance**2) ** 0.5 * np.linalg.inv(hessian)

    #flags for TNC are defined here
    #https://github.com/scipy/scipy/blob/78904d646f6fea3736aa7698394aebd2872e2638/scipy/optimize/tnc/tnc.h#L68-L82
    result.flag = result.status
    #0: good
    #1: error
    #255: shouldn't happen ever
    result.exit = {
      -3: 255, -2: 255, -1: 255,
      0: 1,
      1: 0, 2: 0,
      3: 1,
      4: 255, 5: 255, 6: 255, 7: 255,
    }[result.flag]
    result.dx, result.dy = -result.x
    result.dv = result.fun

    result.R_error_stat = R_error_stat
    result.R_error_syst = R_error_syst
    result.F_error = F_error
    result.covxx = covariancematrix[0,0]
    result.covyy = covariancematrix[1,1]
    result.covxy = covariancematrix[0,1]

    x, y = result.x

    return result

  def __shifted_arrays(self, dx, dy, with_average=False):
    if np.isclose(dx, int(dx)): dx = int(dx)
    if np.isclose(dy, int(dy)): dy = int(dy)

    if dx > 0:
      x1 = abs(dx)
      x2 = 0
    else:
      x1 = 0
      x2 = abs(dx)

    if dy > 0:
      y1 = abs(dy)
      y2 = 0
    else:
      y1 = 0
      y2 = abs(dy)
    #x1 and x2 have dimensions of length

    #or None: https://stackoverflow.com/a/21914093/5228524
    if isinstance(dx, int) and isinstance(dy, int):
      newa = self.a[y1:-y2 or None,x1:-x2 or None]
      newb = self.b[y2:-y1 or None,x2:-x1 or None]
    else:
      newa, newb = shiftimg([self.a, self.b], -dx, -dy, getaverage=False)#dimensions of intensity
      shavex = int(abs(dx)/2)                                            #dimensions of length
      shavey = int(abs(dy)/2)                                            #dimensions of length
      newa = newa[shavey:-shavey or None, shavex:-shavex or None]
      newb = newb[shavey:-shavey or None, shavex:-shavex or None]

    return newa, newb

  def __shifted_array_difference(self, dx, dy):
     newa, newb = self.shifted_arrays(dx, dy)
     return newa - newb

  def __shifted_array_average(self, dx, dy):
     newa, newb = self.shifted_arrays(dx, dy)
     return((newa + newb) / 2)

  def __evalkernel(self, dx, dy, *, nbins=None):
    #dx and dy have dimensions of length
    newa, newb = self.shifted_arrays(dx, dy)
    dd = self.shifted_array_difference(dx, dy)                           #dimensions of intensity

    if nbins is None:
      return np.std(dd)                                                  #dimensions of intensity
    else:
      average = self.shifted_array_average(dx, dy)
      binboundaries = np.quantile(average, np.linspace(0, 1, nbins+1))

      x = []
      y = []
      for low, high in more_itertools.pairwise(binboundaries):
        slice = (low < average) & (average <= high) & (abs(dd) < average)
        if not np.any(slice): continue
        x.append((low+high)/2)
        y.append(np.std(dd[slice]))

      return scipy.interpolate.UnivariateSpline(x, y)

def makespline(x, y, z, knotsx=(), knotsy=()):
  """
  Create a cubic spline fit
  """
  return scipy.interpolate.LSQBivariateSpline(np.ravel(x), np.ravel(y), np.ravel(z), knotsx, knotsy)

def mse(a):
  return np.mean(a**2)

def shiftimg(images, dx, dy, getaverage=True):
  """
  Apply the shift to the two images, using
  a symmetric shift with fractional pixels
  """
  a, b = images

  warpkwargs = {"flags": cv2.INTER_CUBIC, "borderMode": cv2.BORDER_CONSTANT, "dsize": a.T.shape}

  a = cv2.warpAffine(a, np.array([[1, 0,  dx/2], [0, 1,  dy/2]]), **warpkwargs)
  b = cv2.warpAffine(b, np.array([[1, 0, -dx/2], [0, 1, -dy/2]]), **warpkwargs)

  assert a.shape == b.shape == np.shape(images)[1:], (a.shape, b.shape, np.shape(images))

  result = [a, b]
  if getaverage: result.append((a+b)/2)

  return np.array(result)
