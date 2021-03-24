import numba as nb

__vectorizetypelist = [
  nb.float64(nb.float64, nb.float64, nb.float64),
]

@nb.vectorize(__vectorizetypelist)
def micronstopixels(microns, pscale, power):
  return microns * pscale**power
@nb.vectorize(__vectorizetypelist)
def pixelstomicrons(pixels, pscale, power):
  return pixels / pscale**power
