import cv2, functools, logging, numpy as np, scipy.interpolate, scipy.optimize, skimage.filters

logger = logging.getLogger("align")

def computeshift(images, tolerance=0.01):
  widesearcher = ShiftSearcher(images, smoothsigma=4.0)
  finesearcher = ShiftSearcher(images, smoothsigma=1.5)

  #first a 17-wide smooth grid with big steps
  result = widesearcher.search(
    nx=5, xmin=-8, xmax=8,
    ny=5, ymin=-8, ymax=8,
    x0=0, y0=0, tolerance=tolerance,
  )

  done = False

  xmin = ymin = float("inf")
  xmax = ymax = -float("inf")

  i = 3

  logger.debug("%g %g %g", result.dx, result.dy, result.dv)

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
      x0=x0, y0=y0, tolerance=tolerance,
    )
    if prevresult is not None: result.prevresult = prevresult

    i += 1

    logger.debug("%g %g %g %g", result.dx, result.dy, result.dv, result.dverror)

    if i > 5 and abs(
      result.spline(result.dx, result.dy)
      - result.spline(prevresult.dx, prevresult.dy)
    ) < tolerance and abs(
      result.spline(result.dx, result.dy)
      - result.spline(prevresult.prevresult.dx, prevresult.prevresult.dy)
    ):
      break

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
    self.a *= 1/np.sqrt(mse1)
    self.b *= 1/np.sqrt(mse2)

    self.__kernel_kache = {}

    self.evalkernel = np.vectorize(self.__evalkernel)


  def search(self, nx, xmin, xmax, ny, ymin, ymax, x0, y0, tolerance, minimizetolerance=1e-7):
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

    tolerance = max(tolerance, minimizetolerance)

    #create the grid and do brute force evaluations
    gx = np.linspace(xmin, xmax, nx, dtype=int)
    gy = np.linspace(ymin, ymax, ny, dtype=int)
    x, y = np.meshgrid(gx,gy)

    result = scipy.optimize.OptimizeResult()

    result.v = v = self.evalkernel(x, y)
    result.x = x
    result.y = y
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
      tol=minimizetolerance,
      bounds=((xmin, xmax), (ymin, ymax)),
      method="TNC",
    )
    #logger.debug(minimizeresult)
    
    dverror = abs(minimizeresult.fun - self.evalkernel(*minimizeresult.x))

    hessian = np.array([
      [spline(*minimizeresult.x, dx=2, dy=0)[0,0], spline(*minimizeresult.x, dx=1, dy=1)[0,0]],
      [spline(*minimizeresult.x, dx=1, dy=1)[0,0], spline(*minimizeresult.x, dx=0, dy=2)[0,0]],
    ])
    hessianinv = (dverror**2 + minimizetolerance**2) * np.linalg.inv(hessian)

    result.optimizeresult = minimizeresult
    result.flag = result.exit = minimizeresult.status
    result.dx, result.dy = -minimizeresult.x
    result.dv = minimizeresult.fun

    result.dverror = dverror
    result.covxx = hessianinv[0,0]
    result.covyy = hessianinv[1,1]
    result.covxy = hessianinv[0,1]

    x, y = minimizeresult.x

    return result

  def __evalkernel(self, dx, dy):
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

      #or None: https://stackoverflow.com/a/21914093/5228524
      if isinstance(dx, int) and isinstance(dy, int):
        dd = self.a[y1:-y2 or None,x1:-x2 or None] - self.b[y2:-y1 or None,x2:-x1 or None]
      else:
        newa, newb = shiftimg([self.a, self.b], dx, dy, getaverage=False)
        shavex = int(abs(dx)/2)
        shavey = int(abs(dy)/2)
        dd = (newa - newb)[shavey:-shavey or None, shavex:-shavex or None]
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