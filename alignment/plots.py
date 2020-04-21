#!/usr/bin/env python

import itertools, matplotlib.pyplot as plt, more_itertools, networkx as nx, numpy as np, uncertainties.unumpy as unp
from more_itertools import pairwise
from ..utilities import units
from ..utilities.misc import pullhist, weightedaverage, weightedstd

def plotpairwisealignments(alignmentset, *, stitched=False, tags=[1, 2, 3, 4, 6, 7, 8, 9], plotstyling=lambda fig, ax: None, errorbars=True, saveas=None, figurekwargs={}, pull=False, pixelsormicrons=None, pullkwargs={}, pullbinning=None):
  fig = plt.figure(**figurekwargs)
  ax = fig.add_subplot(1, 1, 1)

  vectors = np.array([
    o.result.dxvec - (o.stitchresult if stitched else 0)
    for o in alignmentset.overlaps
    if not o.result.exit
    and o.tag in tags
  ])
  if not errorbars: vectors = unp.nominal_values(vectors)
  if pull:
    if pullbinning is None: pullbinning = np.linspace(-5, 5, 51)
    pullhist(
      vectors[:,0],
      label="$x$ pulls",
      stdinlabel=True,
      alpha=0.5,
      binning=pullbinning,
      **pullkwargs,
    )
    pullhist(
      vectors[:,1],
      label="$y$ pulls",
      stdinlabel=True,
      alpha=0.5,
      binning=pullbinning,
      **pullkwargs,
    )
  else:
    if pullkwargs != {} or pullbinning is not None:
      raise ValueError("Can't provide pull kwargs for a scatter plot")
    if pixelsormicrons is None:
      raise ValueError("Have to provide pixelsormicrons for a scatterplot")
    f = {"pixels": units.pixels, "microns": units.microns}[pixelsormicrons]
    plt.errorbar(
      x=f(units.nominal_values(vectors[:,0])),
      xerr=f(units.std_devs(vectors[:,0])),
      y=f(units.nominal_values(vectors[:,1])),
      yerr=f(units.std_devs(vectors[:,1])),
      fmt='o',
    )

  plotstyling(fig=fig, ax=ax)
  if saveas is None:
    plt.show()
  else:
    plt.savefig(saveas)
    plt.close()

  return vectors

def alignmentshiftprofile(alignmentset, *, deltaxory, vsxory, tag, figurekwargs={}, plotstyling=lambda fig, ax: None, saveas=None, plotsine=False):
  fig = plt.figure(**figurekwargs)
  ax = fig.add_subplot(1, 1, 1)

  xidx = {"x": 0, "y": 1}[vsxory]
  yidx = {"x": 0, "y": 1}[deltaxory]

  class OverlapForProfile:
    def __init__(self, overlap):
      self.overlap = overlap

    def __getattr__(self, attr):
      return getattr(self.overlap, attr)

    @property
    def abspositions(self):
      return self.x1vec[xidx], self.x2vec[xidx]

    @property
    def dx(self):
      return self.result.dxvec[yidx]

  overlaps = [OverlapForProfile(o) for o in alignmentset.overlaps if o.tag == tag]
  allpositions = {o.abspositions for o in overlaps}

  x = []
  y = []
  yerr = []

  for positions in sorted(set(allpositions), key=lambda x: units.pixels(np.mean(x))):
    x.append((positions[0] + positions[1]) / 2)
    dxs = [o.dx for o in overlaps if o.abspositions == positions]
    y.append(units.nominal_value(weightedaverage(dxs)))
    yerr.append(units.nominal_value(weightedstd(dxs)))

  x = np.array(x)
  y = np.array(y)
  yerr = np.array(yerr)

  errorzero = abs(yerr/np.sqrt(sum(y**2)/len(y))) < 1e-10
  errornonzero = ~errorzero

  xwitherror = x[errornonzero]
  ywitherror = y[errornonzero]
  yerrwitherror = yerr[errornonzero]
  xnoerror = x[errorzero]
  ynoerror = y[errorzero]

  plt.errorbar(
    x=units.pixels(xwitherror),
    y=units.pixels(ywitherror),
    yerr=units.pixels(yerrwitherror),
    fmt='o',
    color='b',
  )
  if xnoerror:
    plt.scatter(
      x=units.pixels(xnoerror),
      y=units.pixels(ynoerror),
      facecolors='none',
      edgecolors='b',
    )

  #fit to sine wave
  #for the initial parameter estimation, do an fft
  #can only do fft on evenly spaced data
  # ==> if there are islands, we have to pick the biggest one
  alldeltaxs = {b-a for a, b in more_itertools.pairwise(x)}
  deltax = min(alldeltaxs)
  rtol = 1e-7
  for _ in alldeltaxs:
    units.testing.assert_allclose(_*(1+2*rtol) // deltax, _ / deltax, rtol=rtol)

  chunkstarts = [
    xx for xx in x if not np.any(units.isclose(xx-deltax, x, rtol=rtol))
  ]
  chunkends = [
    xx for xx in x if not np.any(units.isclose(xx+deltax, x, rtol=rtol))
  ]
  biggestchunkstart, biggestchunkend = max(
    itertools.zip_longest(chunkstarts, chunkends),
    key=lambda startend: startend[1]-startend[0],
  )
  biggestchunkxs = np.array([xx for xx in x if biggestchunkstart <= xx <= biggestchunkend])
  biggestchunkys = np.array([yy for xx, yy in zip(x, y) if xx in biggestchunkxs])

  k = np.fft.fftfreq(len(biggestchunkys), deltax)
  f = units.fft.fft(biggestchunkys)

  def cosfunction(xx, amplitude, kk, phase, mean):
    return amplitude * np.cos(np.array(kk*(xx - biggestchunkxs[0]) + phase).astype(float)) + mean

  bestk, bestf = max(zip(k[1:], f[1:]), key=lambda kf: abs(kf[1]))  #[1:]: exclude k=0 term
  initialguess = (
    abs(bestf) / (len(biggestchunkxs) / 2),
    bestk * 2 * np.pi,
    units.angle(bestf),
    np.mean(biggestchunkys)
  )

  p, cov = units.optimize.curve_fit(
    cosfunction, xwitherror, ywitherror, p0=initialguess, sigma=yerrwitherror, absolute_sigma=True,
  )
  p = amplitude, kk, phase, mean = units.correlated_distances(distances=p, covariance=cov)
  print("Average:")
  print(" ", mean)
  try:
    o = overlaps[0]
    expected = ((alignmentset.T - np.identity(2)) @ (o.x1vec - o.x2vec))[yidx]
  except AttributeError:
    pass
  else:
    print(" ", f"(expected from T matrix: {expected})")

  xplot = units.linspace(min(x), max(x), 1000)
  if plotsine:
    #plt.plot(xplot, cosfunction(xplot, *initialguess), color='g')
    plt.plot(units.pixels(xplot), units.pixels(cosfunction(xplot, *units.nominal_values(p))), color='b')

  plotstyling(fig=fig, ax=ax)
  if saveas is None:
    plt.show()
  else:
    plt.savefig(saveas)
    plt.close()

  return x, y, yerr, p

def closedlooppulls(alignmentset, *, tagsequence, binning=np.linspace(-5, 5, 51), quantileforstats=1, verbose=True, stitchresult=None, saveas=None, figurekwargs={}, plotstyling=lambda fig, ax: None):
  dct = {
    1: (-1, -1),
    2: ( 0, -1),
    3: ( 1, -1),
    4: (-1,  0),
    6: ( 1,  0),
    7: (-1,  1),
    8: ( 0,  1),
    9: ( 1,  1),
  }
  totaloffset = sum(np.array(dct[tag]) for tag in tagsequence)
  if np.any(totaloffset):
    raise ValueError(f"please check your tag sequence - it ends with an offset of {tuple(totaloffset)}")

  overlaps = alignmentset.overlaps
  g = alignmentset.overlapgraph()
  xresiduals, yresiduals = [], []
  overlapdict = nx.get_edge_attributes(g, "overlap")

  for o in overlaps:
    if o.result.exit: continue
    if o.tag != tagsequence[-1]: continue
    for path in nx.algorithms.simple_paths.shortest_simple_paths(g, o.p2, o.p1):
      if len(path) < len(tagsequence): continue
      if len(path) > len(tagsequence): break
      path = path[:]
      path.append(path[0]) #make a full circle

      overlaps = [overlapdict[nodepair] for nodepair in pairwise(path)]
      assert overlaps[-1] is o
      tags = [o.tag for o in overlaps]
      if tags != tagsequence: continue

      if any(o.result.exit for o in overlaps): continue
      if stitchresult is not None:
        dxvecs = [stitchresult.dx(o) for o in overlaps]
      else:
        dxvecs = [o.result.dxvec for o in overlaps]

      dxs, dys = zip(*dxvecs)

      if verbose is True or verbose is not False and verbose(path, dxs, dys):
        print(" --> ".join(f"{node:4d}" for node in path))
        for nodepair, dx, dy in zip(pairwise(path), dxs, dys):
          print(f"  {nodepair[0]:4d} --> {nodepair[1]:4d}: {dx:10} {dy:10}")
        print(f"          total: {sum(dxs):10} {sum(dys):10}")

      totaldx = sum(dxs)
      totaldy = sum(dys)
      xresiduals.append(totaldx)
      yresiduals.append(totaldy)

  if verbose:
    print()
    print()
    print()
  fig = plt.figure(**figurekwargs)
  ax = fig.add_subplot(1, 1, 1)
  print("x pulls:")
  pullhist(xresiduals, binning=binning, verbose=True, alpha=0.5, label="$x$ pulls", stdinlabel=True)
  print()
  print()
  print()
  print("y pulls:")
  pullhist(yresiduals, binning=binning, verbose=True, alpha=0.5, label="$y$ pulls", stdinlabel=True)
  plotstyling(fig=fig, ax=ax)

  if saveas is not None:
    plt.savefig(saveas)
    plt.close()

  return xresiduals, yresiduals
