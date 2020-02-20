import cv2, functools, logging, more_itertools, numba as nb, numpy as np, scipy.interpolate, scipy.optimize, skimage.filters, textwrap, uncertainties

logger = logging.getLogger("align")

def computeshift(images):
  """
  https://www.scirp.org/html/8-2660057_43054.htm
  """
  fourier = np.fft.fft2(images)
  crosspower = getcrosspower(fourier)
  invfourier = np.real(np.fft.ifft2(crosspower))
  y, x = np.mgrid[0:invfourier.shape[0],0:invfourier.shape[1]]
  x = np.ravel(x)
  y = np.ravel(y)
  z = np.ravel(invfourier)

  maxidx = np.argmax(z)
  mux0 = x[maxidx]
  muy0 = y[maxidx]
  covxx0 = covyy0 = 1.
  covxy0 = 0.
  A0 = z[maxidx]

  p0 = np.array([mux0, muy0, covxx0, covyy0, covxy0, A0])

  #delete me!
  p0[:] = 250.6668413139962, 2.0840262502716596, 4.673295014296137, 7.522943223827396, 0.0019985244960734096, 0.009531276372970677

  f = functools.partial(vectorizedperiodicdoublegaussian, shape=invfourier.shape)

  mux, muy, covxx, covyy, covxy, A = uncertainties.correlated_values(
    *scipy.optimize.curve_fit(
      f,
      np.array([x, y], order="F"),
      z,
      p0=p0
    )
  )

  logger.info("%s %s %s %s %s %s", mux, muy, covxx, covyy, covxy, A)

  print(
    [mux.n, muy.n],
    uncertainties.covariance_matrix([mux, muy]) + np.array([[covxx.n, covxy.n], [covxy.n, covyy.n]]),
  )

  mux *= -1
  muy *= -1

  dx, dy = uncertainties.correlated_values(
    [mux.n, muy.n],
    uncertainties.covariance_matrix([mux, muy]) + np.array([[covxx.n, covxy.n], [covxy.n, covyy.n]]),
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
            result += doublegaussian(x, mux+i*shape0, muy+j*shape1, invcov, A)
    return result

def vectorizedperiodicdoublegaussian(x, mux, muy, covxx, covyy, covxy, A, shape):
    cov = np.array([[covxx, covxy], [covxy, covyy]])
    invcov = np.linalg.inv(cov)

    result = np.apply_along_axis(periodicdoublegaussian, 0, x, mux, muy, invcov, A, shape)

    print(mux, muy, covxx, covyy, covxy, A, result)
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
