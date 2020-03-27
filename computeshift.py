import cv2, functools, logging, matplotlib.pyplot as plt, more_itertools, numba as nb, numpy as np, scipy.interpolate, scipy.optimize, skimage.feature, skimage.filters, textwrap, uncertainties as unc, uncertainties.unumpy as unp

logger = logging.getLogger("align")

def computeshift(images, *, gputhread=None, gpufftdict=None, windowsize=10, smoothsigma=None, window=None, showsmallimage=False, showbigimage=False, errorfactor=1/2):
  """
  https://www.scirp.org/html/8-2660057_43054.htm
  """
  if smoothsigma is not None:
    images = tuple(skimage.filters.gaussian(image, sigma=smoothsigma, mode = 'nearest') for image in images)
  if window is not None:
    images = tuple(window(image) for image in images)

  if gputhread is not None and gpufftdict is not None :
    images_gpu = tuple(image.astype(np.csingle) for image in images)
    fftc = gpufftdict[images_gpu[0].shape]
    invfourier = crosscorrelation_gpu(images_gpu,gputhread,fftc)
  else :
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
    slice(maxidx[0]-windowsize, maxidx[0]+windowsize+1),
    slice(maxidx[1]-windowsize, maxidx[1]+windowsize+1),
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

  shifted = shiftimg(images, -r.x[0], -r.x[1], clip=False)
  staterror = abs(shifted[0] - shifted[1]) / 2
  #cross correlation evaluated at 0
  error_crosscorrelation = np.sqrt(np.sum(
    staterror**2 * (shifted[0]**2 + shifted[1]**2)
  ))

  covariance = error_crosscorrelation * errorfactor**2 * np.linalg.inv(hessian)

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
def hann(image):
  M, N = image.shape
  hannx = np.hanning(M)
  hanny = np.hanning(N)
  hann = np.outer(hannx, hanny)
  return image * hann

def crosscorrelation_gpu(images,thread,fftc):
  image_devs = tuple(thread.to_device(image) for image in images)
  res_devs   = tuple(thread.empty_like(image_dev) for image_dev in image_devs)
  for resd,imd in zip(res_devs,image_devs) :
    fftc(resd,imd,0)
  crosspower = getcrosspower(tuple(res_dev.get() for res_dev in res_devs))
  cp_dev = thread.to_device(crosspower)
  fftc(cp_dev,cp_dev,1)
  return np.real(cp_dev.get())

def crosscorrelation(images):
  fourier = tuple(np.fft.fft2(image) for image in images)
  crosspower = getcrosspower(fourier)
  invfourier = np.fft.ifft2(crosspower)
  return np.real(invfourier)

@nb.njit
def getcrosspower(fourier):
  return fourier[0] * np.conj(fourier[1])

@nb.njit
def mse(a):
  return np.mean(a*a)

def shiftimg(images, dx, dy, *, clip=True):
  """
  Apply the shift to the two images, using
  a symmetric shift with fractional pixels
  """
  a, b = images
  a = a.astype(float)
  b = b.astype(float)

  warpkwargs = {"flags": cv2.INTER_CUBIC, "borderMode": cv2.BORDER_CONSTANT, "dsize": a.T.shape}

  a = cv2.warpAffine(a, np.array([[1, 0,  dx/2], [0, 1,  dy/2]]), **warpkwargs)
  b = cv2.warpAffine(b, np.array([[1, 0, -dx/2], [0, 1, -dy/2]]), **warpkwargs)

  assert a.shape == b.shape == np.shape(images)[1:], (a.shape, b.shape, np.shape(images))

  if clip:
    ww = 10*(1+int(max(np.abs([dx, dy]))/10))
    clipslice = slice(ww, -ww or None), slice(ww, -ww or None)
  else:
    clipslice = ...

  return np.array([a[clipslice], b[clipslice]])

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
