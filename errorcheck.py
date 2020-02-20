#!/usr/bin/env python

import matplotlib.pyplot as plt, networkx as nx, numpy as np, scipy, uncertainties
from more_itertools import pairwise

def errorcheck(alignmentset, *, tagsequence, binning=np.linspace(-10, 10, 51), quantileforstats=0.99, verbose=True):
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
  g = alignmentset.overlapgraph
  pullsx, pullsy = [], []
  overlapdict = nx.get_edge_attributes(g, "overlap")

  for o in overlaps:
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
      dxdys = [o.result.dxdy for o in overlaps]

      dxs, dys = zip(*dxdys)

      if verbose is True or verbose(path, dxs, dys):
        print(" --> ".join(f"{node:4d}" for node in path))
        for nodepair, dx, dy in zip(pairwise(path), dxs, dys):
          print(f"  {nodepair[0]:4d} --> {nodepair[1]:4d}: {dx:10} {dy:10}")
        print(f"          total: {sum(dxs):10} {sum(dys):10}")

      totaldx = sum(dxs)
      totaldy = sum(dys)
      pullsx.append(totaldx.n / totaldx.s)
      pullsy.append(totaldy.n / totaldy.s)

  if verbose:
    print()
    print()
    print()
  pullsx = np.array(pullsx)
  pullsy = np.array(pullsy)
  quantiles = np.array(sorted(((1-quantileforstats)/2, (1+quantileforstats)/2)))
  minpull, maxpull = np.quantile(np.array([pullsx, pullsy]), quantiles)
  outliersx = len(pullsx[(minpull > pullsx) | (pullsx > maxpull)])
  outliersy = len(pullsy[(minpull > pullsy) | (pullsy > maxpull)])
  pullsx = pullsx[(minpull <= pullsx) & (pullsx <= maxpull)]
  pullsy = pullsy[(minpull <= pullsy) & (pullsy <= maxpull)]
  print("x pulls:")
  plt.hist(pullsx, bins=binning, alpha=0.5)
  print(f"mean of middle {100*quantileforstats}%:   ", uncertainties.ufloat(np.mean(pullsx), scipy.stats.sem(pullsx)))
  print(f"std dev of middle {100*quantileforstats}%:", uncertainties.ufloat(np.std(pullsx), np.std(pullsx) / np.sqrt(2*len(pullsx)-2)))
  print("n outliers: ", outliersx)
  print()
  print()
  print()
  print("y pulls:")
  plt.hist(pullsy, bins=binning, alpha=0.5)
  print(f"mean of middle {100*quantileforstats}%:   ", uncertainties.ufloat(np.mean(pullsy), scipy.stats.sem(pullsy)))
  print(f"std dev of middle {100*quantileforstats}%:", uncertainties.ufloat(np.std(pullsy), np.std(pullsy) / np.sqrt(2*len(pullsy)-2)))
  print("n outliers: ", outliersy)
