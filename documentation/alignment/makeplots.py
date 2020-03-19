#!/usr/bin/env python

import os, matplotlib.pyplot as plt, numpy as np, scipy.interpolate
from ...alignmentset import AlignmentSet

here = os.path.dirname(__file__)
data = os.path.join(here, "..", "..", "test", "data")

rc = {
  "text.usetex": True,
  "text.latex.preamble": [r"\usepackage{amsmath}"],
}

def overlap():
  A = AlignmentSet(data, os.path.join(data, "flatw"), "M21_1")
  A.getDAPI()
  A.readalignments()

  o = A.overlaps[140]
  with plt.rc_context(rc=rc):
    o.showimages(shifted=False, normalize=1000, saveas=os.path.join(here, "overlap-notshifted.pdf"))
    o.showimages(shifted=True, normalize=1000, saveas=os.path.join(here, "overlap-shifted.pdf"))

def maximize1D():
  np.random.seed(123456)
  xx = np.linspace(-5, 5, 11)
  yy = 100 - (xx-0.5)**2 + 0.05 * (xx-0.5)**3 + 2*(np.random.random(xx.shape) - 0.5)

  ymin = min(yy) - (max(yy) - min(yy)) / 10

  spline = scipy.interpolate.UnivariateSpline(xx, yy)
  x = np.linspace(-5, 5, 51)
  y = spline(x)
  maxidx = np.argmax(y)

  r = scipy.optimize.minimize(
    fun=lambda x: -spline(x),
    x0=x[maxidx],
    jac=lambda x: -spline(x, nu=1),
    bounds=((-5, 5),),
    method="TNC"
  )

  with plt.rc_context(rc=rc):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    polynomial, = plt.plot(x, y, color="blue")
    scatter = plt.scatter(xx, yy, color=polynomial.get_color())
    maxline, = plt.plot([r.x, r.x], [-r.fun, ymin], linestyle=":", color="orange")
    maxpoint = plt.scatter(r.x, -r.fun, color=maxline.get_color())

    plt.ylim(bottom=ymin)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.text(r.x - (xmax - xmin) / 50, ymin - (ymax - ymin) / 20, r"$\delta x_\text{max}$", color=maxline.get_color())

    ax.set_xlabel(r"$\delta x$")
    ax.set_ylabel(r"$C(\delta x)$")
    plt.savefig(os.path.join(here, "1Dmaximization.pdf"))
    plt.savefig(os.path.join(here, "1Dmaximizationwitherror.pdf"))

if __name__ == "__main__":
  #overlap()
  maximize1D()
