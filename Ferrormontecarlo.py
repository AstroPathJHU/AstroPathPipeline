#!/usr/bin/env python

import dataclasses, itertools, numpy as np, scipy

from computeshift import makespline
from readtable import writetable

"""
This script is meant to find the factor of 0.5
in step 4, top left of page 3 in
https://www.osti.gov/servlets/purl/934781
That paper dealt with 1D splines, we use 2D,
so the factor might change.  They found the factor
using an ad hoc estimate from Monte Carlo
simulations.  Here we'll do the same.
"""

def makegaussian(meanx, meany, sigmax, sigmay, sigmaxy):
  @np.vectorize
  def gaussian(x, y):
    x -= meanx
    y -= meany
    return np.exp(-0.5 * ((x/sigmax)**2 - (y/sigmay)**2 - (x*y/sigmaxy**2)))
  return gaussian

@dataclasses.dataclass
class MonteCarloResult(object):
  sigmax: float
  sigmay: float
  sigmaxy: float
  unnormalized_F_error: float
  real_error: float

xmin = ymin = -1
xmax = ymax = 1
x, y = np.meshgrid(np.linspace(xmin, xmax, 11), np.linspace(ymin, ymax, 11))

results = []

for sigmax, sigmay, sigmaxy in itertools.product((1, 10, float("inf")), repeat=3):
  gaussian = makegaussian(0, 0, sigmax, sigmay, sigmaxy)
  z = gaussian(x, y)
  spline = makespline(x, y, z)
  Kprimespline = makespline(x, y, z, (0,), (0,))

  maximizeerror = scipy.optimize.differential_evolution(
    func=lambda xy: -abs(spline(*xy) - Kprimespline(*xy))[0,0],
    bounds=((xmin, xmax), (ymin, ymax)),
  )
  unnormalized_F_error = -maximizeerror.fun

  maximizerealerror = scipy.optimize.differential_evolution(
    func=lambda xy: -abs(spline(*xy) - gaussian(*xy))[0,0],
    bounds=((xmin, xmax), (ymin, ymax)),
  )
  real_error = -maximizerealerror.fun

  results.append(MonteCarloResult(sigmax=sigmax, sigmay=sigmay, sigmaxy=sigmaxy, unnormalized_F_error=unnormalized_F_error, real_error=real_error))

writetable("montecarlo.csv", results)
