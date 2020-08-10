import numpy as np
from ..utilities import units
from .plots import plotpairwisealignments, plt

def isotropy(alignmentset, **kwargs):
  vectors = plotpairwisealignments(alignmentset, showplot=False, saveas=None, pull=False, pixelsormicrons="pixels", **kwargs)
  maxfreq = 5
  bins = 48
  cosintegrals = {i: 0 for i in range(maxfreq)}
  sinintegrals = {i: 0 for i in range(maxfreq)}
  plt.hist([np.arctan2(*units.nominal_values(v)) for v in vectors], bins=bins)
  for v in vectors:
    r = 1#units.nominal_values(sum(v**2))
    phi = np.arctan2(*units.nominal_values(v))
    for i in range(maxfreq):
      cosintegrals[i] += r * np.cos(i*phi) / len(vectors) / np.pi
      sinintegrals[i] += r * np.sin(i*phi) / len(vectors) / np.pi
  x = np.linspace(-np.pi, np.pi, 1001)
  y = 0*x
  for i in range(maxfreq):
    print(f"{i:3d}  {cosintegrals[i]:.3f}  {sinintegrals[i]:.3f}")
    y += len(vectors) * (cosintegrals[i] * np.cos(i*x) + sinintegrals[i] * np.sin(i*x)) / (2 if i==0 else 1) * (2*np.pi / bins)
    plt.plot(x, y)
