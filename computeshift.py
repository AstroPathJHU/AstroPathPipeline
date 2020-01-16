import functools, logging, numpy as np, scipy.interpolate, scipy.optimize, skimage.filters

logger = logging.getLogger("align")

def computeshift(images):
  a, b = images

  #first a 17-wide smooth grid with big steps
  LL = 8
  NG = 5
  result = smoothsearch(
    a, b, smoothsigma=4.0,
    nx=5, xmin=-8, xmax=8,
    ny=5, ymin=-8, ymax=8,
    x0=0, y0=0,
  )

  done = False

  while not done:
    prevresult = result
    x0 = int(np.round(-prevresult.dx))
    y0 = int(np.round(-prevresult.dy))
    LL = 4
    NG = 2*LL+1
    result = smoothsearch(
      a, b, smoothsigma=1.5,
      nx=9, xmin=x0-4, xmax=x0+4,
      ny=9, ymin=y0-4, ymax=y0+4,
      x0=x0, y0=y0,
    )
    if prevresult is not None: result.prevresult = prevresult

    if not result.onboundary: done = True

  return result

def smoothsearch(a, b, smoothsigma, nx, xmin, xmax, ny, ymin, ymax, x0, y0, tolerance=1e-7):
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

  #smooth the images
  if smoothsigma != 1:
     a = skimage.filters.gaussian(a, sigma=smoothsigma, mode = 'nearest')
     b = skimage.filters.gaussian(b, sigma=smoothsigma, mode = 'nearest')

  #rescale the intensity
  mse1 = mse(a)
  mse2 = mse(b)
  s = (mse1*mse2)**0.25
  a *= s/np.sqrt(mse1)
  b *= s/np.sqrt(mse2)

  logger.debug("%s %s %s %s %s", mse1, mse2, s, mse(a), mse(b))

  #create the grid and do brute force evaluations
  gx = np.linspace(xmin, xmax, nx)
  gy = np.linspace(ymin, ymax, ny)
  X, Y = np.meshgrid(gx,gy)

  result = scipy.optimize.OptimizeResult()

  result.v = v = eval2v(a, b, X, Y)
  result.x = X
  result.y = Y
  logger.debug("%s %s %s", X, Y, v)
  result.x0 = x0
  result.y0 = y0

  #fit cubic spline to the cost fn
  spline = result.spline = fitS2(X, Y, v)

  #find lowest point for inititalization of the gradient search
  minindices = np.unravel_index(np.argmin(v), v.shape)
  result.xc = xc = float(X[minindices])
  result.yc = yc = float(Y[minindices])

  minx = np.min(X)
  maxx = np.max(X)
  miny = np.min(Y)
  maxy = np.max(Y)

  assert minx == xmin and miny == ymin and maxx == xmax and maxy == ymax

  minimizeresult = scipy.optimize.minimize(
    fun=lambda xy: spline(*xy)[0,0],
    x0=(xc, yc),
    jac=lambda xy: np.array([spline(*xy, dx=1)[0,0], spline(*xy, dy=1)[0,0]]),
    tol=tolerance,
    bounds=((minx, maxx), (miny, maxy)),
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
  result.onboundary = (
    np.isclose(x, minx) or np.isclose(x, maxx)
    or np.isclose(y, miny) or np.isclose(y, maxy)
  )

  return result


def eval2v(A, B, gx, gy):
  """
  Cross-correlate the overlapping segments and
  determine the relative pixel shifts.
    a,c are the two images
    gx, gy are a 1D vector of eval coordinates
  """

  gx = gx.astype(int)
  gy = gy.astype(int)
  return evalkernel(A, B, gx, gy)

@functools.partial(np.vectorize, excluded=(0, 1)) #vectorize over dx and dy
def evalkernel(A,B,dx,dy):
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
    dd = A[y1:-y2 or None,x1:-x2 or None] - B[y2:-y1 or None,x2:-x1 or None]
    return np.std(dd)


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
