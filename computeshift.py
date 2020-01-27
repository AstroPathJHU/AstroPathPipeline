import cv2, functools, logging, numpy as np, scipy.interpolate, scipy.optimize, skimage.filters

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

    logger.debug("%g %g %g %g %g", result.dx, result.dy, result.dv, result.R_error, result.F_error)

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

    self.__kernel_kache = {}

    self.evalkernel = np.vectorize(self.__evalkernel)


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

    minimizeresult = scipy.optimize.minimize(
      fun=lambda xy: spline(*xy)[0,0],            #dimensions of intensity
      x0=(xc, yc),                                #dimensions of length
      jac=lambda xy: np.array([spline(*xy, dx=1)[0,0], spline(*xy, dy=1)[0,0]]),  #dimensions of intensity/length
      tol=minimizetolerance,                      #dimensions of intensity
      bounds=((xmin, xmax), (ymin, ymax)),        #dimensions of length
      method="TNC",
    )

    #calculating error according to https://www.osti.gov/servlets/purl/934781
    #first: R-error from eq. (10)
    Delta_t = 1  #dimensions of length

    #need to estimate sigma_e: error on the spline data points
    #the data points are calculated from evalkernel: standard deviation of (a-b)
    #std dev of the final difference gives an estimate of the intensity error on a or b
    error_on_pixel = self.evalkernel(*minimizeresult.x)   #dimensions of intensity
    """
    \begin{align}
    \mathtt{evalkernel}^2 = K^2 &= \frac{1}{n} \sum_i (a_i - b_i)^2 \\
    (\delta(K^2))^2 &= \frac{1}{n^2} \sum_i 4(a_i-b_i)^2((\delta a_i)^2+(\delta b_i)^2) \\
    &= \frac{8K^2(\mathtt{error\_on\_pixel})^2}{n} \\
    \delta(K^2) &= 2K \sqrt{\frac{2}{n}}(\mathtt{error\_on\_pixel}) \\
    \delta K &= \frac{\delta(K^2)}{2K} \\
    &=\sqrt{\frac{2}{n}}(\mathtt{error\_on\_pixel})
    \end{align}
    """
    sigma_e = np.sqrt(2 / ((self.a.shape[0] - int(abs(minimizeresult.x[0]))) * (self.a.shape[1] - int(abs(minimizeresult.x[1]))))) * error_on_pixel  #dimensions of intensity/length
    kj = []
    deltav = np.zeros(v.shape)
    for idx in np.ndindex(v.shape):
      deltav[idx] = 1
      deltaspline = makespline(x, y, deltav)           #spline(length, length) = dimensionless
      kj.append(deltaspline(*minimizeresult.x))        #dimensionless
      deltav[idx] = 0
    R_error = Delta_t * sigma_e * np.linalg.norm(kj)   #dimensions of intensity
    logger.debug("%g %g %g", error_on_pixel, sigma_e, R_error)

    #F-error from section V
    Kprimespline = makespline(x, y, v, ((xmin+xmax)/2,), ((ymin+ymax)/2,))
    maximizeerror = scipy.optimize.differential_evolution(
      func=lambda xy: -abs(spline(*xy) - Kprimespline(*xy))[0,0],
      bounds=((xmin, xmax), (ymin, ymax)),
    )
    F_error = -0.5 * maximizeerror.fun

    hessian = np.array([
      [spline(*minimizeresult.x, dx=2, dy=0)[0,0], spline(*minimizeresult.x, dx=1, dy=1)[0,0]],
      [spline(*minimizeresult.x, dx=1, dy=1)[0,0], spline(*minimizeresult.x, dx=0, dy=2)[0,0]],
    ])   #dimensions of intensity / length^2
    hessianinv = (F_error**2 + R_error**2 + minimizetolerance**2) * np.linalg.inv(hessian)

    result.optimizeresult = minimizeresult
    result.flag = result.exit = minimizeresult.status
    result.dx, result.dy = -minimizeresult.x
    result.dv = minimizeresult.fun

    result.R_error = R_error
    result.F_error = F_error
    result.covxx = hessianinv[0,0]
    result.covyy = hessianinv[1,1]
    result.covxy = hessianinv[0,1]

    x, y = minimizeresult.x

    return result

  def __evalkernel(self, dx, dy):
    #dx and dy have dimensions of length
    if (dx, dy) not in self.__kernel_kache:
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
        dd = self.a[y1:-y2 or None,x1:-x2 or None] - self.b[y2:-y1 or None,x2:-x1 or None]  #dimensions of intensity
      else:
        newa, newb = shiftimg([self.a, self.b], dx, dy, getaverage=False)  #dimensions of intensity
        shavex = int(abs(dx)/2)                                            #dimensions of length
        shavey = int(abs(dy)/2)                                            #dimensions of length
        dd = (newa - newb)[shavey:-shavey or None, shavex:-shavex or None] #dimensions of intensity
      self.__kernel_kache[dx, dy] = np.std(dd)                             #dimensions of intensity

    return self.__kernel_kache[dx, dy]                                     #dimensions of intensity

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
