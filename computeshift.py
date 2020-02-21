import cv2, functools, logging, more_itertools, numba as nb, numpy as np, scipy.interpolate, scipy.optimize, skimage.filters, textwrap, uncertainties as unc, uncertainties.unumpy as unp

logger = logging.getLogger("align")

def computeshift(images, smoothsigma=1.5, window=lambda images: hann(images)):
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

  maxidx = np.unravel_index(np.argmax(z, axis=None), z.shape)
  mux = x[maxidx]
  muy = y[maxidx]
  covxx = covyy = 1.
  covxy = 0.
  A = z[maxidx]

  dx = dy = unc.ufloat(0, 9999)
  windowsize = 0

  p0 = unp.nominal_values(np.array([mux, muy, covxx, covyy, covxy, A]))

  while np.sqrt(np.trace(unc.covariance_matrix([dx, dy]))) > windowsize / 2:
    windowsize += 10

    slc = (
      slice(maxidx[0]-windowsize, maxidx[0]+windowsize),
      slice(maxidx[1]-windowsize, maxidx[1]+windowsize),
    )
    xx = x[slc]
    yy = y[slc]
    zz = z[slc]

    xx = np.ravel(xx)
    yy = np.ravel(yy)
    zz = np.ravel(zz)

    #p0 = unp.nominal_values(np.array([mux, muy, covxx, covyy, covxy, A]))

    #delete me!
    #p0[:] = 250.6668413139962, 2.0840262502716596, 4.673295014296137, 7.522943223827396, 0.0019985244960734096, 0.009531276372970677

    f = functools.partial(vectorizedperiodicdoublegaussian, shape=invfourier.shape)

    p, cov = scipy.optimize.curve_fit(
      f,
      np.array([xx, yy], order="F"),
      zz,
      p0=p0,
    )
    mux, muy, covxx, covyy, covxy, A = unc.correlated_values(p, cov)

    logger.info("%s %s %s %s %s %s %d", mux, muy, covxx, covyy, covxy, A, windowsize)

    print(
      [mux.n, muy.n],
      unc.covariance_matrix([mux, muy]) + np.array([[covxx.n, covxy.n], [covxy.n, covyy.n]]),
    )

    dx, dy = unc.correlated_values(
      [-mux.n, -muy.n],
      unc.covariance_matrix([mux, muy]) + np.array([[covxx.n, covxy.n], [covxy.n, covyy.n]]),
    )

  while dx.n >= invfourier.shape[1] / 2: dx -= invfourier.shape[1]
  while dx.n < -invfourier.shape[1] / 2: dx += invfourier.shape[1]
  while dy.n >= invfourier.shape[0] / 2: dy -= invfourier.shape[0]
  while dy.n < -invfourier.shape[0] / 2: dy += invfourier.shape[0]

  return OptimizeResult(
    dx=dx,
    dy=dy,
  )

@nb.njit
def hann(images):
  M, N = images.shape[1:]
  hannx = np.hanning(M)
  hanny = np.hanning(N)
  hann = np.outer(hannx, hanny)
  return images * hann

@nb.njit
def getcrosspower(fourier):
  product = fourier[0] * np.conj(fourier[1])
  crosspower = product / np.abs(product)
  return crosspower

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
    return repr(v)

  def __repr__(self):
    if self.keys():
      m = max(map(len, list(self.keys()))) + 1
      return '\n'.join([k.rjust(m) + ': ' + self.__formatvforrepr(v, m)
                        for k, v in sorted(self.items())])
    else:
      return self.__class__.__name__ + "()"
