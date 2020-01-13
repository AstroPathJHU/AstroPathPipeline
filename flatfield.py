import logging, numpy as np, scipy.optimize

logger = logging.getLogger("align")

def meanimage(images, logmsg=""):
  logger.info(logmsg)

  img = np.mean(images, axis=0)

  #todo: figure out what this code is doing

  positiveindices = img > 0
  meanofmeanimage = np.mean(img[positiveindices])
  img /= meanofmeanimage
  img[~positiveindices] = 1.0

  m, n = img.shape
  x, y = np.meshgrid(range(1, n+1),range(1, m+1))
  xdata = x.flatten()
  ydata = y.flatten()
  zdata = img.flatten()

  #least squares fit:
  #problem has to be set up to get an approximate solution to Ax=b
  #A = matrix with columns of xdata^m ydata^n
  #x = vector of coeffs (what we want)
  #b = zdata

  A = makequadraticpolynomial(xdata, ydata).T
  b = zdata

  fitresult = scipy.optimize.lsq_linear(A, b)
  coeffs = fitresult.x
  fitresult.function = lambda x, y: np.tensordot(coeffs, makequadraticpolynomial(x, y), axes=(0, 0))
  fitresult.flatfield = fitresult.function(x, y)
  fitresult.rawflatfield = img
  fitresult.ratio  = img / fitresult.flatfield

  return fitresult

def makequadraticpolynomial(x, y):
  """
  Take x and y, which are variable values.
  Turn them into an array of the terms that will
  appear in a quadratic polynomial.
  The order of terms is 1, x, x^2, y, xy, y^2
  """
  assert np.shape(x) == np.shape(y)
  return np.array([
    np.ones(np.shape(x)),
    x,
    x**2,
    y,
    x*y,
    y**2,
  ])

def createfitflat(img):
  """
  Least square fit for abcdefg:
  img = a + bx + cx^2 + dy + exy + fy^2
  """
