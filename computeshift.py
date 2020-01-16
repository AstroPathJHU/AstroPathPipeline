import functools, logging, numpy as np, scipy.interpolate, scipy.optimize, skimage.filters

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

  while True:
    prevresult = result
    x0 = int(np.round(-prevresult.dx))
    y0 = int(np.round(-prevresult.dy))

    oldxmin, oldxmax, oldymin, oldymax = xmin, xmax, ymin, ymax

    xmin = int(min(xmin, -prevresult.dx - 3))
    ymin = int(min(ymin, -prevresult.dy - 3))
    xmax = int(max(xmax, -prevresult.dx + 3))
    ymax = int(max(ymax, -prevresult.dy + 3))

    if (oldxmin, oldxmax, oldymin, oldymax) == (xmin, xmax, ymin, ymax):
      break

    result = finesearcher.search(
      nx=xmax-xmin+1, xmin=xmin, xmax=xmax,
      ny=ymax-ymin+1, ymin=ymin, ymax=ymax,
      x0=x0, y0=y0,
    )
    if prevresult is not None: result.prevresult = prevresult

  logger.debug(
    "had to compute %d * %d = %d points",
    xmax-xmin+1, ymax-ymin+1, (xmax-xmin+1) * (ymax-ymin+1)
  )

  return result

class ShiftSearcher:
  def __init__(self, images, smoothsigma):
    self.images = images
    self.smoothsigma = smoothsigma

    a, b = images

    #smooth the images
    if smoothsigma != 1:
      self.a = skimage.filters.gaussian(a, sigma=smoothsigma, mode = 'nearest')
      self.b = skimage.filters.gaussian(b, sigma=smoothsigma, mode = 'nearest')

    #rescale the intensity
    mse1 = mse(self.a)
    mse2 = mse(self.b)
    s = (mse1*mse2)**0.25
    self.a *= s/np.sqrt(mse1)
    self.b *= s/np.sqrt(mse2)

    logger.debug("%s %s %s %s %s", mse1, mse2, s, mse(a), mse(b))

    self.__kernel_kache = {}

    self.evalkernel = np.vectorize(self.__evalkernel)


  def search(self, nx, xmin, xmax, ny, ymin, ymax, x0, y0, tolerance=1e-7):
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

    #create the grid and do brute force evaluations
    gx = np.linspace(xmin, xmax, nx, dtype=int)
    gy = np.linspace(ymin, ymax, ny, dtype=int)
    x, y = np.meshgrid(gx,gy)

    result = scipy.optimize.OptimizeResult()

    result.v = v = self.evalkernel(x, y)
    result.x = x
    result.y = y
    logger.debug("%s %s %s", x, y, v)
    result.x0 = x0
    result.y0 = y0

    #fit cubic spline to the cost fn
    spline = result.spline = fitS2(x, y, v)

    #find lowest point for inititalization of the gradient search
    minindices = np.unravel_index(np.argmin(v), v.shape)
    result.xc = xc = float(x[minindices])
    result.yc = yc = float(y[minindices])

    minimizeresult = scipy.optimize.minimize(
      fun=lambda xy: spline(*xy)[0,0],
      x0=(xc, yc),
      jac=lambda xy: np.array([spline(*xy, dx=1)[0,0], spline(*xy, dy=1)[0,0]]),
      tol=tolerance,
      bounds=((xmin, xmax), (ymin, ymax)),
      method="TNC",
    )
    logger.debug(minimizeresult)

    hessian = np.array([
      [spline(*minimizeresult.x, dx=2, dy=0)[0,0], spline(*minimizeresult.x, dx=1, dy=1)[0,0]],
      [spline(*minimizeresult.x, dx=1, dy=1)[0,0], spline(*minimizeresult.x, dx=0, dy=2)[0,0]],
    ])
    hessianinv = tolerance * np.linalg.inv(hessian)

    result.optimizeresult = minimizeresult
    result.flag = result.exit = minimizeresult.status
    result.dx, result.dy = -minimizeresult.x
    result.dv = minimizeresult.fun

    result.tolerance = tolerance
    result.covxx = hessianinv[0,0]
    result.covyy = hessianinv[1,1]
    result.covxy = hessianinv[0,1]

    x, y = minimizeresult.x

    return result

  def __evalkernel(self, dx, dy):
    if (dx, dy) not in self.__kernel_kache:
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

      #or None: https://stackoverflow.com/a/21914093/5228524
      dd = self.a[y1:-y2 or None,x1:-x2 or None] - self.b[y2:-y1 or None,x2:-x1 or None]
      result = self.__kernel_kache[dx, dy] = np.std(dd)

    return self.__kernel_kache[dx, dy]

def fitS2(x, y, z):
  """
  Create a cubic spline fit
  """
  xdata = x.flatten()
  ydata = y.flatten()
  zdata = z.flatten()

  return scipy.interpolate.SmoothBivariateSpline(xdata, ydata, zdata)

def mse(a):
  return np.mean(a**2)
