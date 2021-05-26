import numpy as np, scipy.optimize

from ...utilities.misc import dummylogger

def meanimage(images, *, logger=dummylogger):
  """
  Find the flat field, which is the average intensity
  over all the images, parameterized as a quadratic
  polynomial in x and y.  Returns an OptimizeResult
  from scipy with some extra terms added.  To flatten
  the image, divide by fitresult.flatfield
  """

  logger.info("meanimage")

  if len(np.shape(images)) == 3:
    #image index, height, width
    pass
  elif len(np.shape(images)) == 4:
    #image index, layer, height, width
    result = scipy.optimize.OptimizeResult()
    result.flatfield = [
      meanimage(layer).flatfield
      for layer in np.transpose(images, (1, 0, 2, 3))
    ]
    return result
  else:
    raise ValueError(f"Can't handle shape {np.shape(images)}")

  img = np.mean(images, axis=0)

  positiveindices = img > 0
  meanofmeanimage = np.mean(img[positiveindices])
  img /= meanofmeanimage
  img[~positiveindices] = 1.0

  m, n = img.shape
  x, y = np.meshgrid(np.arange(1, n+1) / n, np.arange(1, m+1) / m)
  xdata = np.ravel(x)
  ydata = np.ravel(y)
  zdata = np.ravel(img)

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
