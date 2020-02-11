#!/usr/bin/env python

import matplotlib.pyplot as plt, networkx as nx, numpy as np, scipy, uncertainties
from more_itertools import pairwise

def errorcheck(alignmentset, *, tagsequence):
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

      dxdys = [o.result.dxdy for o in overlaps]
      if any(not np.all(np.isfinite(o.result.covariance)) for o in overlaps): continue

      dxs, dys = zip(*dxdys)

      print(" --> ".join(f"{node:3d}" for node in path))
      for nodepair, dx, dy in zip(pairwise(path), dxs, dys):
        print(f"  {nodepair[0]:3d} --> {nodepair[1]:3d}: {dx:10} {dy:10}")
      print(f"        total: {sum(dxs):10} {sum(dys):10}")

      totaldx = sum(dxs)
      totaldy = sum(dys)
      pullsx.append(totaldx.n / totaldx.s)
      pullsy.append(totaldy.n / totaldy.s)

  print()
  print()
  print()
  print("x pulls:")
  plt.hist(pullsx)
  print("mean:   ", uncertainties.ufloat(np.mean(pullsx), scipy.stats.sem(pullsx)))
  print("std dev:", uncertainties.ufloat(np.std(pullsx), np.std(pullsx) / np.sqrt(2*len(pullsx)-2)))
  print()
  print()
  print()
  print("y pulls:")
  plt.hist(pullsy)
  print("mean:   ", uncertainties.ufloat(np.mean(pullsy), scipy.stats.sem(pullsy)))
  print("std dev:", uncertainties.ufloat(np.std(pullsy), np.std(pullsy) / np.sqrt(2*len(pullsy)-2)))
