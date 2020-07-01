import itertools, scipy.optimize
from ..core import _power, _pscale, distances, pixels

def curve_fit(f, xdata, ydata, p0, sigma=None, **kwargs):
  pscale = {_pscale(_)[()] for _ in itertools.chain(p0, xdata, ydata, sigma if sigma is not None else [])}
  if len(pscale) > 1: pscale.discard(None)
  assert len(pscale) == 1
  pscale = pscale.pop()

  xpower = {_power(_)[()] for _ in xdata}
  if len(xpower) > 1: xpower.discard(None)
  assert len(xpower) == 1
  xpower = xpower.pop()
  
  ypower = {_power(_)[()] for _ in itertools.chain(ydata, sigma if sigma is not None else [])}
  if len(ypower) > 1: ypower.discard(None)
  assert len(ypower) == 1
  ypower = ypower.pop()

  ppowers = [_power(_) for _ in p0]
  covpowers = [[a + b for b in ppowers] for a in ppowers]

  xpixels = pixels(xdata, power=xpower)
  ypixels = pixels(ydata, power=ypower)
  p0pixels = pixels(p0, power=ppowers)
  sigmapixels = pixels(sigma, power=ypower) if sigma is not None else None

  def fpixels(xpixels, *ppixels):
    x = distances(pixels=xpixels, power=xpower, pscale=pscale)
    p = distances(pixels=ppixels, power=ppowers, pscale=pscale)
    result = f(x, *p)
    resultpixels = pixels(result, power=ypower, pscale=pscale)
    return resultpixels

  ppixels, covpixels = scipy.optimize.curve_fit(fpixels, xpixels, ypixels, p0=p0pixels, sigma=sigmapixels, **kwargs)

  p = distances(pixels=ppixels, pscale=pscale, power=ppowers)
  cov = distances(pixels=covpixels, pscale=pscale, power=covpowers)

  return p, cov
