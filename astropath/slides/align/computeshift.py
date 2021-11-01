import cv2, matplotlib.pyplot as plt, numba as nb, numpy as np, scipy.interpolate, scipy.optimize, skimage.feature, skimage.filters, textwrap, uncertainties as unc

def computeshift(images, *, gputhread=None, gpufftdict=None, windowsize=10, smoothsigma=None, window=None, showsmallimage=False, savesmallimage=None, showbigimage=False, savebigimage=None, errorfactor=1/4, staterrorimages=None, usemaxmovementcut=True, mindistancetootherpeak=None, checkpositivedefinite=True):
  """
  Compute the relative shift between two images by maximizing the cross correlation.
  The cross correlation is computed using the FFT.

  International Journal of Medical Physics, Clinical Engineering and Radiation Oncology
  Vol.3 No.1(2014), Paper ID 43054, 7 pages
  DOI:10.4236/ijmpcero.2014.31008

  gputhread, gpufftdict: a gpu thread and dict of compiled FFT functions, if using the GPU (default: None)
  windowsize: window around the maximum to use for fitting to compute subpixel shifts (default: 10). Can also be a tuple of (windowsizex, windowsizey).
  smoothsigma: width to use for gaussian smearing (default: None)
  window: window to apply to the image after smearing (default: None)
  errorfactor: scale the computed error by this factor (default: 1/4; see the LaTeX document in the documentation folder)
  staterrorimages: precomputed statistical error on the images (default: None; the statistical error is computed as difference between the two images after alignment)
  usemaxmovementcut: report the alignment as failed if the shift is more than 10% of the image size
  mindistancetootherpeak: report the alignment as failed if there's another peak in the cross correlation within this distance of the main peak (default: np.max(windowsize))
  checkpositivedefinite: report the alignment as failed if the Hessian matrix of the 2nd degree polynomial fitted to the peak is not positive definite

  #the remaining arguments are for debugging
  showsmallimage: show the zoomed cross correlation image
  savesmallimage: filename to save the zoomed cross correlation image
  showbigimage: show the cross correlation image
  savebigimage: filename to save the cross correlation image
  """
  #smooth the images
  if smoothsigma is not None:
    images = tuple(skimage.filters.gaussian(image, sigma=smoothsigma, mode = 'nearest') for image in images)
  #apply a window to the images
  if window is not None:
    images = tuple(window(image) for image in images)
  #calculate the cross correlation between the images
  use_gpu = gputhread is not None and gpufftdict is not None
  if use_gpu :
    images_gpu = tuple(image.astype(np.csingle) for image in images)
    fftc = gpufftdict[images_gpu[0].shape]
    invfourier = crosscorrelation_gpu(images_gpu,gputhread,fftc)
  else :
    invfourier = crosscorrelation(images)

  y, x = np.mgrid[0:invfourier.shape[0],0:invfourier.shape[1]]
  z = invfourier

  #find the maximum integer value of the cross correlation
  maxidx = np.unravel_index(np.argmax(np.abs(z)), z.shape)
  rollby = np.array(z.shape)//2 - maxidx

  #roll to get the peak in the middle
  x = np.roll(x, rollby[0], axis=0)
  y = np.roll(y, rollby[0], axis=0)
  z = np.roll(z, rollby[0], axis=0)
  x = np.roll(x, rollby[1], axis=1)
  y = np.roll(y, rollby[1], axis=1)
  z = np.roll(z, rollby[1], axis=1)

  maxidx = tuple((maxidx + rollby) % z.shape)

  #change coordinate system, so 0 is in the middle
  x[x > x.shape[1]/2] -= x.shape[1]
  y[y > y.shape[0]/2] -= y.shape[0]

  try:
    windowsizex, windowsizey = windowsize
  except TypeError:
    windowsizex = windowsizey = windowsize
  windowsize = np.array([windowsizex, windowsizey])

  #zoom into around the maximum
  slc = (
    slice(max(maxidx[0]-windowsizey, 0), maxidx[0]+windowsizey+1),
    slice(max(maxidx[1]-windowsizex, 0), maxidx[1]+windowsizex+1),
  )
  xx = x[slc]
  yy = y[slc]
  zz = z[slc]

  if showbigimage or savebigimage:
    plt.imshow(z, extent=[x.min(), x.max(), y.min(), y.max()])
    plt.xlabel(r"$\delta x$")
    plt.ylabel(r"$\delta y$", labelpad=-5)
    if savebigimage:
      plt.savefig(savebigimage)
    if showbigimage:
      plt.show()
    if savebigimage:
      plt.close()
  if showsmallimage or savesmallimage:
    plt.imshow(zz, extent=[xx.min(), xx.max(), yy.min(), yy.max()])
    plt.xlabel(r"$\delta x$")
    plt.ylabel(r"$\delta y$", labelpad=-5)
    if savesmallimage:
      plt.savefig(savesmallimage)
    if showsmallimage:
      plt.show()
    if savesmallimage:
      plt.close()

  xx = doravel(xx)
  yy = doravel(yy)
  zz = doravel(zz)

  #fit to a spline and maximize the subpixel shift
  knotsx = ()
  knotsy = ()
  spline = scipy.interpolate.LSQBivariateSpline(xx, yy, zz, knotsx, knotsy)
  def f(*args, **kwargs): return spline(*args, **kwargs)[0,0]

  r = scipy.optimize.minimize(
    fun=lambda xy: -f(*xy),
    x0=(x[maxidx], y[maxidx]),
    jac=lambda xy: np.array([-f(*xy, dx=1), -f(*xy, dy=1)]),
    bounds=((x[maxidx]-windowsizex, x[maxidx]+windowsizex), (y[maxidx]-windowsizey, y[maxidx]+windowsizey)),
    method="TNC",
  )
  crosscorr = -r.fun

  #calculate the error on the shift from the 2nd derivative of the spline
  hessian = -np.array([
    [f(*r.x, dx=2, dy=0), f(*r.x, dx=1, dy=1)],
    [f(*r.x, dx=1, dy=1), f(*r.x, dx=0, dy=2)],
  ])

  shifted = shiftimg(images, -r.x[0], -r.x[1], clip=False, use_gpu=use_gpu)
  if staterrorimages is None:
    staterror0 = staterror1 = abs(shifted[0] - shifted[1])
  else:
    staterror0, staterror1 = staterrorimages
  #cross correlation evaluated at 0
  error_crosscorrelation = np.sqrt(np.sum(
    (staterror0 * shifted[1])**2 + (staterror1 * shifted[0])**2
  ))

  covariance = 2 * error_crosscorrelation * errorfactor**2 * np.linalg.inv(hessian)

  exit = 0
  dx, dy = unc.correlated_values(
    -r.x,
    covariance
  )

  #various error codes:
  #  if there are other significant peaks in the cross correlation
  if mindistancetootherpeak is None:
    mindistancetootherpeak = np.max(windowsize)
  otherbigindices = skimage.feature.corner_peaks(z, min_distance=mindistancetootherpeak, threshold_abs=z[maxidx] - 3*error_crosscorrelation, threshold_rel=0)
  for idx in otherbigindices:
    if np.all(idx == maxidx): continue
    if np.all(np.abs(idx - maxidx) < windowsize): continue
    dx = unc.ufloat(0, 9999.)
    dy = unc.ufloat(0, 9999.)
    exit = 1
    break

  #  if the covariance matrix is not positive definite
  if checkpositivedefinite and not np.all(np.linalg.eig(covariance)[0] > 0):
    dx = unc.ufloat(0, 9999.)
    dy = unc.ufloat(0, 9999.)
    exit = 2

  #  if the shift is more than 10% of the size of the overlap
  if usemaxmovementcut and np.sqrt(np.sum(r.x**2) >= np.sqrt(np.sum(np.array(x.shape)**2))) / 10:
    dx = unc.ufloat(0, 9999.)
    dy = unc.ufloat(0, 9999.)
    exit = 3

  return OptimizeResult(
    dx=dx,
    dy=dy,
    crosscorrelation=unc.ufloat(crosscorr, error_crosscorrelation),
    exit=exit,
    spline=spline,
  )

def crosscorrelation_gpu(images,thread,fftc):
  """
  Calculate the cross correlation using the GPU
  """
  image_devs = tuple(thread.to_device(image) for image in images)
  res_devs   = tuple(thread.empty_like(image_dev) for image_dev in image_devs)
  for resd,imd in zip(res_devs,image_devs) :
    fftc(resd,imd,0)
  crosspower = getcrosspower(tuple(res_dev.get() for res_dev in res_devs))
  cp_dev = thread.to_device(crosspower)
  fftc(cp_dev,cp_dev,1)
  return np.real(cp_dev.get())

def crosscorrelation(images):
  """
  Calculate the cross correlation without the GPU
  """
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

@nb.njit
def doravel(a):
  return np.ravel(a)

def shiftimg(images, dx, dy, *, clip=True, use_gpu=False, shiftwhich=None):
  """
  Apply the shift to the two images, using
  a symmetric shift with fractional pixels
  """
  dx = float(dx)
  dy = float(dy)

  shifta = {
    0: 2,
    None: 1,
    1: 0,
  }[shiftwhich]
  shiftb = {
    0: 0,
    None: 1,
    1: 2,
  }[shiftwhich]

  a, b = images
  if shifta: a = a.astype(np.float32)
  if shiftb: b = b.astype(np.float32)

  warpkwargs = {"flags": cv2.INTER_CUBIC, "borderMode": cv2.BORDER_CONSTANT, "dsize": a.T.shape}

  if use_gpu :
    if shifta: a = cv2.UMat(a)
    if shiftb: b = cv2.UMat(b)

  if shifta: a = cv2.warpAffine(a, np.array([[1, 0,  dx*shifta/2], [0, 1,  dy*shifta/2]]), **warpkwargs)
  if shiftb: b = cv2.warpAffine(b, np.array([[1, 0, -dx*shiftb/2], [0, 1, -dy*shiftb/2]]), **warpkwargs)

  if use_gpu :
    if shifta: a = a.get()
    if shiftb: b = b.get()

  assert a.shape == b.shape == np.shape(images)[1:], (a.shape, b.shape, np.shape(images))

  if clip:
    ww = 10*(1+int(max(np.abs([dx, dy]))/10))
    clipslice = slice(ww*shifta, -ww*shiftb or None), slice(ww*shifta, -ww*shiftb or None)
  else:
    clipslice = ...

  return np.array([a[clipslice], b[clipslice]])

class OptimizeResult(scipy.optimize.OptimizeResult):
  """
  Like a scipy OptimizeResult, but if one of the values is another OptimizeResult,
  it will format it nicely
  """
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
