import collections, numpy as np, pathlib
from ...utilities import units
from .plots import plotpairwisealignments, plt

def isotropy(alignsample, maxfreq=5, bins=24, showplot=None, saveas=None, figurekwargs={}, plotstyling=lambda fig, ax: None, **kwargs):
  fig = plt.figure(**figurekwargs)
  ax = fig.add_subplot(1, 1, 1)
  vectors = plotpairwisealignments(alignsample, showplot=False, saveas=None, pull=False, errorbars=False, pixelsormicrons="pixels", **kwargs)
  cosintegrals = np.zeros(shape=maxfreq)
  sinintegrals = np.zeros(shape=maxfreq)
  rs = []
  phis = []
  for v in vectors:
    rs.append(sum(v**2)**.5)
    phis.append(units.np.arctan2(*reversed(v)))
  plt.hist(phis, bins=bins)
  #plt.hist(phis, weights=rs/alignsample.onepixel, bins=bins)
  for r, phi in zip(rs, phis):
    for i in range(maxfreq):
      cosintegrals[i] += np.cos(i*phi) / np.pi
      sinintegrals[i] += np.sin(i*phi) / np.pi
  x = np.linspace(-np.pi, np.pi, 1001)
  y = 0*x
  for i in range(maxfreq):
    alignsample.printlogger.info(f"{i:3d}  {cosintegrals[i]:.3f}  {sinintegrals[i]:.3f}")
    y += (cosintegrals[i] * np.cos(i*x) + sinintegrals[i] * np.sin(i*x)) / (2 if i==0 else 1) * (2*np.pi / bins)
    plt.plot(x, y, label=f"{i}")

  for s1 in 1, -1:
    for s2 in 1, -1:
      corner = units.np.arctan2(s1*alignsample.fheight, s2*alignsample.fwidth)
      plt.axvline(corner, 0, .65, color='r')
      diagonal = units.np.arctan2(s1, s2)
      plt.axvline(diagonal, 0, .65, color='b')

  plotstyling(fig, ax)

  if showplot is None:
    showplot = saveas is None

  if showplot:
    plt.show()
  if saveas is not None:
    plt.savefig(saveas)
  if not showplot:
    plt.close()

  return cosintegrals, sinintegrals, np.mean(np.array(rs)**2)**.5

def stitchingisotropy(alignsample, cornerfractions=np.linspace(0, 1, 26), showplot=None, saveas=None, figurekwargs={}, plotstyling=lambda fig, ax: None, **kwargs):
  saveas = pathlib.Path(saveas)
  if saveas is not None:
    assert "{tag}" in saveas.name
    assert "{isotropyorRMS}" in saveas.name
  ampfourier = collections.defaultdict(lambda: [])
  RMS = collections.defaultdict(lambda: [])
  for cornerfraction in cornerfractions:
    alignsample.logger.info("%g", cornerfraction)
    result = alignsample.stitch(saveresult=False, scaleedges=1-cornerfraction, scalecorners=cornerfraction)
    alignsample.applystitchresult(result)
    for tag in 1, 2, 3, 4:
      cosfourier, sinfourier, rms = isotropy(alignsample, stitched=True, showplot=False, tags=[tag], **kwargs)
      ampfourier[tag].append((cosfourier**2 + sinfourier**2) ** .5)
      RMS[tag].append(rms)

  for tag in 1, 2, 3, 4, "all":
    for thing in "isotropy", "rms":
      if tag == "all" and thing == "isotropy": continue
      fig = plt.figure(**figurekwargs)
      ax = fig.add_subplot(1, 1, 1)

      if thing == "isotropy":
        ampfourier[tag] = np.array(ampfourier[tag])
        for i, amps in enumerate(ampfourier[tag].T):
          plt.scatter(cornerfractions, amps, label=i)
      elif thing == "rms":
        if tag == "all":
          y = sum(np.array(RMS[tag])**2 for tag in (1, 2, 3, 4))**.5
        else:
          y = np.array(RMS[tag])
        plt.scatter(cornerfractions, y/alignsample.onepixel)

      plotstyling(fig, ax, thing)

      if showplot is None:
        showplot = saveas is None

      if showplot:
        plt.show()
      if saveas is not None:
        plt.savefig(str(saveas).format(tag=tag, isotropyorRMS=thing))
      if not showplot:
        plt.close()

  return ampfourier, RMS
