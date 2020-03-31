#!/usr/bin/env python

import matplotlib.pyplot as plt, networkx as nx, numpy as np, uncertainties.unumpy as unp
from more_itertools import pairwise
from .utilities import pullhist, savefig

def plotpairwisealignments(alignmentset, *, stitched=False, tags=[1, 2, 3, 4, 6, 7, 8, 9], plotstyling=lambda fig, ax: None, errorbars=True, saveas=None, figurekwargs={}, pull=False, pullkwargs={}, pullbinning=None):
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
    plt.errorbar(
      x=unp.nominal_values(vectors[:,0]),
      xerr=unp.std_devs(vectors[:,0]),
      y=unp.nominal_values(vectors[:,1]),
      yerr=unp.std_devs(vectors[:,1]),
      fmt='o',
    )

  plotstyling(fig=fig, ax=ax)
  if saveas is None:
    plt.show()
  else:
    savefig(saveas)
    plt.close()

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

      if any(not np.all(o.result.covariance < 9998) for o in overlaps): continue
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
