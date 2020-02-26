import cv2, functools, logging, matplotlib.pyplot as plt, more_itertools, numba as nb, numpy as np, scipy.interpolate, scipy.optimize, skimage.feature, skimage.filters, textwrap, uncertainties as unc, uncertainties.unumpy as unp

logger = logging.getLogger("align")

def computeshift(images, *, windowsize=10, smoothsigma=None, window=None, showsmallimage=False, showbigimage=False):
  """
  https://www.scirp.org/html/8-2660057_43054.htm
  """
  if smoothsigma is not None:
    images = skimage.filters.gaussian(images, sigma=smoothsigma, mode = 'nearest')
  if window is not None:
    images = window(images)

  invfourier = crosscorrelation(images)

  y, x = np.mgrid[0:invfourier.shape[0],0:invfourier.shape[1]]
  z = invfourier

  #roll to get the peak in the middle
  x = np.roll(x, x.shape[0]//2, axis=0)
  y = np.roll(y, y.shape[0]//2, axis=0)
  z = np.roll(z, z.shape[0]//2, axis=0)
  x = np.roll(x, x.shape[1]//2, axis=1)
  y = np.roll(y, y.shape[1]//2, axis=1)
  z = np.roll(z, z.shape[1]//2, axis=1)

  #change coordinate system, so 0 is in the middle
  x[x > x.shape[1]/2] -= x.shape[1]
  y[y > y.shape[0]/2] -= y.shape[0]

  maxidx = np.unravel_index(np.argmax(np.abs(z)), z.shape)

  slc = (
    slice(maxidx[0]-windowsize, maxidx[0]+windowsize),
    slice(maxidx[1]-windowsize, maxidx[1]+windowsize),
  )
  xx = x[slc]
  yy = y[slc]
  zz = z[slc]

  if showbigimage: plt.imshow(z)
  if showsmallimage: plt.imshow(zz)

  xx = np.ravel(xx)
  yy = np.ravel(yy)
  zz = np.ravel(zz)

  knotsx = ()
  knotsy = ()
  spline = scipy.interpolate.LSQBivariateSpline(xx, yy, zz, knotsx, knotsy)
  def f(*args, **kwargs): return spline(*args, **kwargs)[0,0]

  r = scipy.optimize.minimize(
    fun=lambda xy: -f(*xy),
    x0=(x[maxidx], y[maxidx]),
    jac=lambda xy: np.array([-f(*xy, dx=1), -f(*xy, dy=1)]),
    bounds=((x[maxidx]-windowsize, x[maxidx]+windowsize), (y[maxidx]-windowsize, y[maxidx]+windowsize)),
    method="TNC",
  )

  hessian = -np.array([
    [f(*r.x, dx=2, dy=0), f(*r.x, dx=1, dy=1)],
    [f(*r.x, dx=1, dy=1), f(*r.x, dx=0, dy=2)],
  ])

  shifted = shiftimg(images, -r.x[0], -r.x[1], getaverage=False)
  staterrorspline = statisticalerrorspline(shifted, nbins=20)
  #cross correlation evaluated at 0
  error_crosscorrelation = np.sqrt(np.sum(
    (staterrorspline(shifted[0]) * shifted[1]) ** 2
  + (shifted[0] * staterrorspline(shifted[1])) ** 2
  ))

  logger.info("%g %g %g", z[maxidx], error_crosscorrelation, error_crosscorrelation / z[maxidx])

  covariance = error_crosscorrelation * np.linalg.inv(hessian)

  exit = 0

  otherbigindices = skimage.feature.corner_peaks(z, min_distance=windowsize, threshold_abs=z[maxidx] - error_crosscorrelation)
  for idx in otherbigindices:
    if np.all(idx == maxidx): continue
    displacement = idx - maxidx
    dist = np.linalg.norm(displacement)
    angle = np.arctan2(displacement[1], displacement[0])
    rotationmatrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

    covariance = rotationmatrix @ covariance @ rotationmatrix.T
    covariance += [[dist**2, 0], [0, 0]]
    covariance = rotationmatrix.T @ covariance @ rotationmatrix
    exit = 1

  dx, dy = unc.correlated_values(
    -r.x,
    covariance
  )

  return OptimizeResult(
    dx=dx,
    dy=dy,
    exit=exit,
    spline=spline,
  )

@nb.njit
def hann(images):
  M, N = images.shape[1:]
  hannx = np.hanning(M)
  hanny = np.hanning(N)
  hann = np.outer(hannx, hanny)
  return images * hann

def crosscorrelation(images):
  fourier = np.fft.fft2(images)
  crosspower = getcrosspower(fourier)
  invfourier = np.real(np.fft.ifft2(crosspower))
  return invfourier

@nb.njit
def getcrosspower(fourier):
  return fourier[0] * np.conj(fourier[1])

def statisticalerrorspline(images, nbins):
  difference = images[0] - images[1]
  average = (images[0]+images[1])/2
  binboundaries = np.quantile(average, np.linspace(0, 1, nbins+1))
  x = []
  y = []
  for low, high in more_itertools.pairwise(binboundaries):
    slice = (low < average) & (average <= high)
    if not np.any(slice): continue
    x.append((low+high)/2)
    y.append(mse(difference[slice])**.5)
  return scipy.interpolate.UnivariateSpline(x, y)

@nb.njit
def mse(a):
  return np.mean(a*a)

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


class OptimizeResult(scipy.optimize.OptimizeResult):
  def __formatvforrepr(self, v, m):
    if isinstance(v, OptimizeResult):
      return "\n" + textwrap.indent(repr(v), ' '*m)
    if isinstance(v, list) and all(isinstance(thing, OptimizeResult) for thing in v):
      return "[\n" + "\n\n".join(textwrap.indent(repr(thing), ' '*m) for thing in v) + "\n" + " "*m + "]"
    return repr(v)

  def __repr__(self):
    if self.keys():
      m = max(map(len, list(self.keys()))) + 1
      return '\n'.join([k.rjust(m) + ': ' + self.__formatvforrepr(v, m)
                        for k, v in sorted(self.items())])
    else:
      return self.__class__.__name__ + "()"
