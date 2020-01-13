import logging, numpy as np, scipy.optimize

logger = logging.getLogger("align")

def meanimage(images, logmsg=""):
  logger.info(logmsg)

  result = np.mean(images, axis=0)

  #todo: figure out what this code is doing

  positiveindices = result > 0
  meanofmeanimage = np.mean(result[positiveindices])
  result /= meanofmeanimage
  result[~positiveindices] = 1.0

  m, n = result.shape
  x, y = np.meshgrid(range(n),range(m))
  fitresult = createfitflat(x,y,result)

  fitresult.flatfield = fitresult.function(x, y)
  fitresult.rawflatfield = result
  fitresult.ratio  = result / fitresult.flatfield

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

def createfitflat(x, y, img):
  """
  Least square fit for abcdefg:
  img = a + bx + cx^2 + dy + exy + fy^2
  """
  assert x.shape == y.shape == img.shape
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

  return fitresult

