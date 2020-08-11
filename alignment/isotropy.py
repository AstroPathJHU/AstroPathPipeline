import numpy as np
from ..utilities import units
from .plots import plotpairwisealignments, plt

def isotropy(alignmentset, maxfreq=5, bins=24, showplot=True, **kwargs):
  vectors = plotpairwisealignments(alignmentset, showplot=False, saveas=None, pull=False, errorbars=False, pixelsormicrons="pixels", **kwargs)
  cosintegrals = np.zeros(shape=maxfreq)
  sinintegrals = np.zeros(shape=maxfreq)
  rs = []
  phis = []
  for v in vectors:
    rs.append(1)
    #rs.append(sum(v**2))
    phis.append(np.arctan2(*v))
  plt.hist(phis, weights=rs, bins=bins)
  for r, phi in zip(rs, phis):
    for i in range(maxfreq):
      cosintegrals[i] += r * np.cos(i*phi) / np.pi
      sinintegrals[i] += r * np.sin(i*phi) / np.pi
  x = np.linspace(-np.pi, np.pi, 1001)
  y = 0*x
  for i in range(maxfreq):
    print(f"{i:3d}  {cosintegrals[i]:.3f}  {sinintegrals[i]:.3f}")
    y += (cosintegrals[i] * np.cos(i*x) + sinintegrals[i] * np.sin(i*x)) / (2 if i==0 else 1) * (2*np.pi / bins)
    plt.plot(x, y)

  if showplot:
    plt.show()
  else:
    plt.close()

  return cosintegrals, sinintegrals

def stitchingisotropy(alignmentset, cornerfractions=np.linspace(0, 1, 101), **kwargs):
  ampfourier = []
  for cornerfraction in cornerfractions:
    result = alignmentset.stitch(saveresult=False, scaleedges=1-cornerfraction, scalecorners=cornerfraction, **kwargs)
    alignmentset.applystitchresult(result)
    cosfourier, sinfourier = isotropy(alignmentset, stitched=True, showplot=False)
    ampfourier.append((cosfourier**2 + sinfourier**2) ** .5)
  ampfourier = np.array(ampfourier)
  for amps in ampfourier.T:
    plt.scatter(cornerfractions, amps)
  plt.show()
  return ampfourier
