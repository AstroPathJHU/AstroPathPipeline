import numpy as np
from .core import distances, pixels, UnitsError

def fft(a, *args, **kwargs):
  fftpixels = np.fft.fft(pixels(a, power=None), *args, **kwargs)

  pscale = {_._pscale for _ in np.ravel(a) if _ and _._pscale is not None}
  if not pscale: return fftpixels
  if len(pscale) > 1: raise UnitsError(f"a has multiple different pscales {pscale}")
  pscale = pscale.pop()

  power = {_._power for _ in np.ravel(a) if _ and _._power is not None}
  if len(power) > 1: raise UnitsError(f"a has multiple different powers {power}")
  power = power.pop()

  return distances(pixels=fftpixels, pscale=pscale, power=power, defaulttozero=True)
