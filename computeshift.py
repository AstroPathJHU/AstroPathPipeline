import numpy as np, scipy.interpolate, scipy.optimize

def computeshift(images):
  a, b = images;

  #first a 17-wide smooth grid with big steps
  LL = 8
  NG = 5
  firstresult = smoothsearch(a,b,4.0,LL,NG,0,0);
  x0 = int(np.round(-f.dx))
  y0 = int(np.round(-f.dy))

  #fine grid with single step
  LL = 4;
  NG = 2*LL+1;
  result = smoothSearch(a,b,1.5,LL,NG,x0,y0);
  result.firstresult = firstresult;
  return result

def smoothSearch(a,b,WW,LL,NG,x0,y0):
  """
  Take the two images a, b, and find their relative shifts.
  a and b are the two images, WW is the smoothing length,
  (-LL,LL) is the grid range, NG is the grid size, and
  x0,y0 is the relative shift between the centers.
  The function fits a bicubic spline to the grids, then does
  a gradient search on the interpolated function for the
  minimum.  Returns a struct containing the final shift,
  and the the grid points for debugging.
  """

  #smooth the images
  if WW != 1:
     a = skimage.filters.gaussian(a, sigma=WW, mode = 'nearest', truncate=2.0)
     b = skimage.filters.gaussian(b, sigma=WW, mode = 'nearest', truncate=2.0)

   #rescale the intensity
   mse1 = mse(a)
   mse2 = mse(b)
   s = (mse1*mse2)**0.25
   a *= s/sqrt(mse1)
   b *= s/sqrt(mse2)

   #create the grid and do brute force evaluations
   gx = np.linspace(x0-LL,x0+LL,NG)
   gy = np.linspace(y0-LL,y0+LL,NG)
   X, Y = np.meshgrid(gx,gy);

   result = OptimizeResult()

   result.v = v = eval2v(a, b, X, Y)
   result.x = X
   result.y = Y
   result.x0 = x0;
   result.y0 = y0;

   #fit cubic spline to the cost fn
   spline = result.spline = fitS2(X, Y, v);

   #find lowest point for inititalization of the gradient search
   minindices = np.argmin(v)
   result.xc = xc = X[minindices]
   result.yc = yc = Y[minindices]

   minimizeresult = scipy.optimize.minimize(
     fun=lambda xy: spline(*xy),
     x0=(xc, yc),
     jac=lambda xy: [spline(*xy, dx=1), spline(*xy, dy=1)],
     hess=lambda xy: [
       [spline(*xy, dx=2, dy=0), spline(*xy, dx=1, dy=1)],
       [spline(*xy, dx=1, dy=1), spline(*xy, dx=0, dy=2)],
     ],
     tol=3e-3,
     bounds=((min(X), max(X)), (min(Y), max(Y))),
     method="TNC",
   )

   result.optimizeresult = minimizeresult
   result.flag = result.exit = flag
   result.dx, result.dy = -minimizeresult.x
   result.dv = minimizeresult.fun

   return result


def eval2v(A, B, gx, gy):
  """
  Cross-correlate the overlapping segments and
  determine the relative pixel shifts.
    a,c are the two images
    gx, gy are a 1D vector of eval coordinates
  """

  v = np.array([evalkernel(A, B, x, y) for x, y in zip(gx, gy)])

  return v

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
  return np.std(a)**2

