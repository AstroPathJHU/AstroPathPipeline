import cv2, functools, logging, matplotlib.pyplot as plt, more_itertools, numba as nb, numpy as np, scipy.interpolate, scipy.optimize, skimage.feature, skimage.filters, textwrap, uncertainties as unc, uncertainties.unumpy as unp

logger = logging.getLogger("align")

def computeshift(images, *, windowsize=10, smoothsigma=None, window=lambda images: hann(images), localmax_min_distance=20, localmax_threshold_rel=.5, localmax_windowsize=20, showsmallimage=False, showbigimage=False):
  """
  https://www.scirp.org/html/8-2660057_43054.htm
  """
  if smoothsigma is not None:
    images = skimage.filters.gaussian(images, sigma=smoothsigma, mode = 'nearest')
  if window is not None:
    images = window(images)
  fourier = np.fft.fft2(images)
  crosspower = getcrosspower(fourier)
  invfourier = np.real(np.fft.ifft2(crosspower))

  y, x = np.mgrid[0:invfourier.shape[0],0:invfourier.shape[1]]
  z = invfourier

  x = np.roll(x, x.shape[0]//2, axis=0)
  y = np.roll(y, y.shape[0]//2, axis=0)
  z = np.roll(z, z.shape[0]//2, axis=0)
  x = np.roll(x, x.shape[1]//2, axis=1)
  y = np.roll(y, y.shape[1]//2, axis=1)
  z = np.roll(z, z.shape[1]//2, axis=1)
  #roll to get the peak in the middle

  #estimate gaussian mean and center
  labels = (
      (x < localmax_windowsize//2)
    | (x > invfourier.shape[1]-localmax_windowsize//2)
  ) & (
      (y < localmax_windowsize//2)
    | (y > invfourier.shape[0]-localmax_windowsize//2)
  )
  maxindices = skimage.feature.peak_local_max(z, min_distance=localmax_min_distance, threshold_rel=localmax_threshold_rel, labels=labels)
  results = []
  for maxidx in maxindices:
    maxidx = tuple(maxidx)
    try:
      mux = x[maxidx]
      muy = y[maxidx]
      A = z[maxidx]

      #estimate gaussian width
      bigpoints = np.argwhere(z>=np.max(z)/2)
      distances = np.linalg.norm(bigpoints-maxidx, axis=1)
      covxx = covyy = max(distances[distances <= windowsize * 2**.5])**2
      covxy = 0.

      p0 = unp.nominal_values(np.array([mux, muy, covxx, covyy, covxy, A]))

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

      f = functools.partial(vectorizedperiodicdoublegaussian, shape=invfourier.shape)

      xfit = np.array([xx, yy], order="F")
      p, cov = scipy.optimize.curve_fit(
        f,
        xfit,
        zz,
        p0=p0,
      )
      mux, muy, covxx, covyy, covxy, A = unc.correlated_values(p, cov)

      logger.info("%s %s %s %s %s %s %d", mux, muy, covxx, covyy, covxy, A, windowsize)

      dx, dy = unc.correlated_values(
        [-mux.n, -muy.n],
        unc.covariance_matrix([mux, muy]) + np.array([[covxx.n, covxy.n], [covxy.n, covyy.n]]),
      )

      while dx.n >= invfourier.shape[1] / 2: dx -= invfourier.shape[1]
      while dx.n < -invfourier.shape[1] / 2: dx += invfourier.shape[1]
      while dy.n >= invfourier.shape[0] / 2: dy -= invfourier.shape[0]
      while dy.n < -invfourier.shape[0] / 2: dy += invfourier.shape[0]

      chi2 = sum((f(xfit, *p) - zz) ** 2)

      results.append(OptimizeResult(
        mux=mux,
        muy=muy,
        covxx=covxx,
        covyy=covyy,
        covxy=covxy,
        A=A,
        p0=p0,
        dx=dx,
        dy=dy,
        chi2=chi2,
      ))
    except Exception as e:
      dx, dy = unc.correlated_values([0, 0], [[9999, 0], [0, 9999]])
      results.append(OptimizeResult(
        exception=e,
      ))

  def resultevaluator(result):
    if hasattr(result, "exception"): return "bad", 0
    cov = unc.covariance_matrix([result.dx, result.dy])
    if np.trace(cov) ** .5 >= sum(invfourier.shape): return "bad", 1
    return "ok", -result.chi2

  result = max(results, key=resultevaluator)
  if hasattr(result, "exception"): raise result.exception
  results.remove(result)
  result.otherresults = results
  return result

@nb.njit
def hann(images):
  M, N = images.shape[1:]
  hannx = np.hanning(M)
  hanny = np.hanning(N)
  hann = np.outer(hannx, hanny)
  return images * hann

@nb.njit
def getcrosspower(fourier):
  return fourier[0] * np.conj(fourier[1])

@nb.njit
def doublegaussian(x, mux, muy, invcov, A):
    mu = np.array([mux, muy])
    shifted = x - mu

    return A * np.exp(-0.5 * shifted @ invcov @ shifted)

@nb.njit
def periodicdoublegaussian(x, mux, muy, invcov, A, shape):
    result = 0
    shape0, shape1 = shape
    for i in -1, 0, 1:
        for j in -1, 0, 1:
            result += doublegaussian(x, mux+i*shape1, muy+j*shape0, invcov, A)
    return result

def vectorizedperiodicdoublegaussian(x, mux, muy, covxx, covyy, covxy, A, shape):
    cov = np.array([[covxx, covxy], [covxy, covyy]])
    invcov = np.linalg.inv(cov)

    result = np.apply_along_axis(periodicdoublegaussian, 0, x, mux, muy, invcov, A, shape)

    return result

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
