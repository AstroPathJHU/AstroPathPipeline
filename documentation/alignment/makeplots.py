#!/usr/bin/env python

import argparse, functools, os, matplotlib.patches as patches, matplotlib.pyplot as plt, numpy as np, scipy.interpolate
from ...alignmentset import AlignmentSet
from ...utilities import savefig

here = os.path.dirname(__file__)
data = os.path.join(here, "..", "..", "test", "data")

rc = {
  "text.usetex": True,
  "text.latex.preamble": [r"\usepackage{amsmath}"],
  "font.size": 20,
  "figure.subplot.bottom": 0.12,
}

@functools.lru_cache()
def alignmentset(*, dapi=False):
  if dapi:
    A = alignmentset()
    A.getDAPI()
    return A

  A = AlignmentSet(data, os.path.join(data, "flatw"), "M21_1")
  A.readalignments()
  A.readstitchresult()
  return A

def overlap():
  A = alignmentset(dapi=True)
  o = A.overlaps[140]
  with plt.rc_context(rc=rc):
    o.showimages(shifted=False, normalize=1000, ticks=True, saveas=os.path.join(here, "overlap-notshifted.pdf"))
    o.showimages(shifted=True, normalize=1000, ticks=True, saveas=os.path.join(here, "overlap-shifted.pdf"))

def xcorrelation():
  A = alignmentset(dapi=True)
  o = A.overlaps[140]
  with plt.rc_context(rc=rc):
    o.align(savebigimage=os.path.join(here, "overlap-xcorrelation.pdf"), alreadyalignedstrategy="overwrite", debug=True)

  o = A.overlaps[203]
  with plt.rc_context(rc=rc):
    o.showimages(shifted=False, normalize=100, ticks=True, saveas=os.path.join(here, "overlap-bad.pdf"))
    o.align(savebigimage=os.path.join(here, "overlap-xcorrelation-bad.pdf"), alreadyalignedstrategy="overwrite", debug=True)

def maximize1D():
  np.random.seed(123456)
  xx = np.linspace(-5, 5, 11)
  yy = 100 - (xx-1.5)**2 + 0.05 * (xx-1.5)**3 + 2*(np.random.random(xx.shape) - 0.5)

  ymin = min(yy) - (max(yy) - min(yy)) / 10

  deltaC = 10

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
  xbest = r.x
  Cbest = -r.fun
  Cminus = Cbest - deltaC
  hessian = -spline(r.x, nu=2)
  covariance = deltaC / (1/2 * hessian)
  deltax = covariance ** .5
  xminus = xbest-deltax
  xplus = xbest+deltax

  with plt.rc_context(rc=rc):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    polynomial, = plt.plot(x, y, color="blue")
    scatter = plt.scatter(xx, yy, color=polynomial.get_color())
    maxline, = plt.plot([xbest, xbest], [Cbest, ymin], linestyle=":", color="orange")
    maxpoint = plt.scatter(xbest, Cbest, color=maxline.get_color())

    plt.ylim(bottom=ymin)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xrange = xmax-xmin
    yrange = ymax-ymin
    deltaxtext = plt.text(xbest - xrange/50, ymin - yrange/40, r"$\delta x_\text{max}$", color=maxline.get_color(), horizontalalignment="center", verticalalignment="top")

    ax.set_xlabel(r"$\delta x$")
    ax.set_ylabel(r"$C(\delta x)$")
    savefig(os.path.join(here, "1Dmaximization.pdf"))

    maxpoint.remove()
    maxline.remove()
    deltaxtext.remove()
    deltaCline, = plt.plot([xbest, xbest], [Cbest, Cminus], linestyle=":", color=maxline.get_color())
    sigmaCtext = plt.text(xbest - xrange/100, Cbest - 5*deltaC/8, r"$\sigma_C$", color=deltaCline.get_color(), horizontalalignment="right", verticalalignment="center")
    deltaxline, = plt.plot([xminus, xplus], [Cminus, Cminus], linestyle=":", color="fuchsia")
    deltaxbrackets, = plt.plot([xminus, xbest, xplus], [Cminus - yrange/20]*3, linestyle="-", color=deltaxline.get_color())
    deltaxbracketends = [
      plt.plot([x, x], [Cminus - yrange/20 - yrange/50, Cminus - yrange/20 + yrange/50], linestyle="-", color=deltaxbrackets.get_color())
      for x in [xminus, xbest, xplus]
    ]
    sigmaxtexts = [
      plt.text(xbest + sign * deltax/2, Cminus - yrange/20 - yrange/40, r"$\sigma_{\delta x_\text{max}}$", color=deltaxbrackets.get_color(), horizontalalignment="center", verticalalignment="top")
      for sign in (-1, 1)
    ]
    savefig(os.path.join(here, "1Dmaximizationwitherror.pdf"))

    plt.close(fig)

def islands():
  A = alignmentset()
  with plt.rc_context(rc=rc):
    plt.imshow(A.image())
    plt.xticks([])
    plt.yticks([])
    savefig(os.path.join(here, "islands.pdf"))
    plt.close()

def alignmentresults():
  A = alignmentset()
  def plotstyling(fig, ax):
    plt.xlabel("$\delta x$")
    plt.ylabel("$\delta y$")
    plt.xlim(left=-5, right=5)
    plt.ylim(bottom=-5, top=5)
    ax.set_aspect("equal", "box")

  kwargs = {
    "plotstyling": plotstyling,
    "figurekwargs": {"figsize": (6, 6)}
  }
  with plt.rc_context(rc=rc):
    for tag in 1, 2, 3, 4:
      A.plotresults(tags=[tag], saveas=os.path.join(here, f"alignment-result-{tag}.pdf"), **kwargs)
      A.plotresults(tags=[tag], stitched=True, saveas=os.path.join(here, f"stitch-result-{tag}.pdf"), **kwargs)

def scanning():
  with plt.rc_context(rc=rc):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(0, 500)
    plt.ylim(500, 0)
    for y in range(0, 500, 100):
      for x in range(0, 500, 100):
        ax.add_patch(patches.Rectangle((x, y), 100, 100, linewidth=1, edgecolor='r', facecolor='none'))
        if x == 400:
          if y != 400:
            plt.arrow(x+50, y+50, -400, 100, width=3, length_includes_head=True)
        else:
          plt.arrow(x+50, y+50, 100, 0, width=3, length_includes_head=True)
    savefig(os.path.join(here, "scanning.pdf"))
    plt.close()

def squarepulls(*, bki):
  with plt.rc_context(rc=rc):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(0, 500)
    plt.ylim(500, 0)
    for y in range(0, 500, 100):
      for x in range(0, 500, 100):
        ax.add_patch(patches.Rectangle((x, y), 100, 100, linewidth=1, edgecolor='r', facecolor='none'))
    x1, y1 = 255, 250
    x2, y2 = 352, 247
    x3, y3 = 349, 353
    x4, y4 = 247, 350
    x5, y5 = 252, 248

    arrows = [
      plt.arrow(x1, y1, x2-x1, y2-y1, width=3, length_includes_head=True)
      plt.arrow(x2, y2, x3-x2, y3-y2, width=3, length_includes_head=True)
      plt.arrow(x3, y3, x4-x3, y4-y3, width=3, length_includes_head=True)
      plt.arrow(x4, y4, x5-x4, y5-y4, width=3, length_includes_head=True)
      plt.arrow(x5, y5, x1-x5, y1-y5, width=3, length_includes_head=True)
    ]

    savefig(os.path.join(here, "squarepulldiagram.pdf"))

    for a in arrows: a.remove()

    x2, y2, x3, y3 = x3, y3, x2+200, y2
    x4 -= 100

    arrows = [
      plt.arrow(x1, y1, x2-x1, y2-y1, width=3, length_includes_head=True)
      plt.arrow(x2, y2, x3-x2, y3-y2, width=3, length_includes_head=True)
      plt.arrow(x3, y3, x4-x3, y4-y3, width=3, length_includes_head=True)
      plt.arrow(x4, y4, x5-x4, y5-y4, width=3, length_includes_head=True)
      plt.arrow(x5, y5, x1-x5, y1-y5, width=3, length_includes_head=True)
    ]

    savefig(os.path.join(here, "diamondpulldiagram.pdf"))

    plt.close()

if __name__ == "__main__":
  class EqualsEverything:
    def __eq__(self, other): return True
  p = argparse.ArgumentParser()
  g = p.add_mutually_exclusive_group()
  g.add_argument("--all", action="store_const", dest="which", const=EqualsEverything(), default=EqualsEverything())
  g.add_argument("--maximize", action="store_const", dest="which", const="maximize")
  g.add_argument("--overlap", action="store_const", dest="which", const="overlap")
  g.add_argument("--xcorrelation", "--cross-correlation", action="store_const", dest="which", const="xcorrelation")
  g.add_argument("--islands", action="store_const", dest="which", const="islands")
  g.add_argument("--alignmentresults", action="store_const", dest="which", const="alignmentresults")
  g.add_argument("--scanning", action="store_const", dest="which", const="scanning")
  args = p.parse_args()

  if args.which == "maximize":
    maximize1D()
  if args.which == "overlap":
    overlap()
  if args.which == "xcorrelation":
    xcorrelation()
  if args.which == "islands":
    islands()
  if args.which == "alignmentresults":
    alignmentresults()
  if args.which == "scanning":
    scanning()
