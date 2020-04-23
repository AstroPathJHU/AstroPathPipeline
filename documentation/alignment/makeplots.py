#!/usr/bin/env python

import argparse, collections, functools, os, matplotlib.patches as patches, matplotlib.pyplot as plt, numpy as np, scipy.interpolate
from ...alignment.plots import alignmentshiftprofile, closedlooppulls, plotpairwisealignments
from ...alignment.alignmentset import AlignmentSet
from ...utilities import units

here = os.path.dirname(__file__)
data = os.path.join(here, "..", "..", "test", "data")

rc = {
  "text.usetex": True,
  "text.latex.preamble": [r"\usepackage{amsmath}\usepackage{siunitx}"],
  "font.size": 20,
  "figure.subplot.bottom": 0.12,
}

@functools.lru_cache()
def alignmentset(*, root1=data, root2=os.path.join(data, "flatw"), samp="M21_1", dapi=False):
  if dapi:
    A = alignmentset(root1=root1, root2=root2, samp=samp)
    A.getDAPI()
    return A

  A = AlignmentSet(root1, root2, samp)
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
    plt.scatter(xx, yy, color=polynomial.get_color())
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
    plt.savefig(os.path.join(here, "1Dmaximization.pdf"))

    maxpoint.remove()
    maxline.remove()
    deltaxtext.remove()
    deltaCline, = plt.plot([xbest, xbest], [Cbest, Cminus], linestyle=":", color=maxline.get_color())
    plt.text(xbest - xrange/100, Cbest - 5*deltaC/8, r"$\sigma_C$", color=deltaCline.get_color(), horizontalalignment="right", verticalalignment="center")
    deltaxline, = plt.plot([xminus, xplus], [Cminus, Cminus], linestyle=":", color="fuchsia")
    deltaxbrackets, = plt.plot([xminus, xbest, xplus], [Cminus - yrange/20]*3, linestyle="-", color=deltaxline.get_color())
    [
      plt.plot([x, x], [Cminus - yrange/20 - yrange/50, Cminus - yrange/20 + yrange/50], linestyle="-", color=deltaxbrackets.get_color())
      for x in [xminus, xbest, xplus]
    ]
    [
      plt.text(xbest + sign * deltax/2, Cminus - yrange/20 - yrange/40, r"$\sigma_{\delta x_\text{max}}$", color=deltaxbrackets.get_color(), horizontalalignment="center", verticalalignment="top")
      for sign in (-1, 1)
    ]
    plt.savefig(os.path.join(here, "1Dmaximizationwitherror.pdf"))

    plt.close(fig)

def islands():
  A = alignmentset()
  with plt.rc_context(rc=rc):
    plt.imshow(A.image())
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(here, "islands.pdf"))
    plt.close()

def alignmentresults():
  A = alignmentset()
  def plotstyling(fig, ax):
    plt.xlabel("$\delta x$ (pixels)")
    plt.ylabel("$\delta y$ (pixels)", labelpad=-10)
    plt.xlim(left=-5, right=5)
    plt.ylim(bottom=-5, top=5)
    plt.subplots_adjust(bottom=0.2, left=0.15)
    ax.set_aspect("equal", "box")

  kwargs = {
    "plotstyling": plotstyling,
    "pixelsormicrons": "pixels",
    "figurekwargs": {"figsize": (3, 3)},
  }
  with plt.rc_context(rc=rc):
    for tag in 1, 2, 3, 4:
      plotpairwisealignments(A, tags=[tag], saveas=os.path.join(here, f"alignment-result-{tag}.pdf"), **kwargs)
      plotpairwisealignments(A, tags=[tag], stitched=True, saveas=os.path.join(here, f"stitch-result-{tag}.pdf"), **kwargs)

def scanning():
  with plt.rc_context(rc=rc):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(0, 500)
    plt.ylim(500, 0)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    for y in range(0, 500, 100):
      for x in range(0, 500, 100):
        ax.add_patch(patches.Rectangle((x, y), 100, 100, linewidth=1, edgecolor='r', facecolor='none'))
        if x == 400:
          if y != 400:
            plt.arrow(x+50, y+50, -400, 100, width=3, length_includes_head=True)
        else:
          plt.arrow(x+50, y+50, 100, 0, width=3, length_includes_head=True)
    plt.savefig(os.path.join(here, "scanning.pdf"))
    plt.close()

def squarepulls(*, bki):
  with plt.rc_context(rc=rc):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(0, 500)
    plt.ylim(500, 0)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    for y in range(0, 500, 100):
      for x in range(0, 500, 100):
        ax.add_patch(patches.Rectangle((x, y), 100, 100, linewidth=1, edgecolor='r', facecolor='none'))
    x1, y1 = 270, 140
    x2, y2 = 332, 157
    x3, y3 = 349, 243
    x4, y4 = 247, 230
    x5, y5 = 252, 158

    arrows = [
      plt.arrow(x1, y1, x2-x1, y2-y1, width=3, length_includes_head=True),
      plt.arrow(x2, y2, x3-x2, y3-y2, width=3, length_includes_head=True),
      plt.arrow(x3, y3, x4-x3, y4-y3, width=3, length_includes_head=True),
      plt.arrow(x4, y4, x5-x4, y5-y4, width=3, length_includes_head=True),
      plt.arrow(x5, y5, x1-x5, y1-y5, width=3, length_includes_head=True, facecolor="orange"),
    ]

    plt.savefig(os.path.join(here, "squarepulldiagram.pdf"))

    for a in arrows: a.remove()

    x2, y2, x3, y3 = x3, y3, x2-100, y2+200
    x4 -= 100

    arrows = [
      plt.arrow(x1, y1, x2-x1, y2-y1, width=3, length_includes_head=True),
      plt.arrow(x2, y2, x3-x2, y3-y2, width=3, length_includes_head=True),
      plt.arrow(x3, y3, x4-x3, y4-y3, width=3, length_includes_head=True),
      plt.arrow(x4, y4, x5-x4, y5-y4, width=3, length_includes_head=True),
      plt.arrow(x5, y5, x1-x5, y1-y5, width=3, length_includes_head=True, facecolor="orange"),
    ]

    plt.savefig(os.path.join(here, "diamondpulldiagram.pdf"))

    plt.close()

    if bki:
      def plotstyling(*, fig, ax, squareordiamond):
        plt.xlabel(rf"$\delta(x,y)^\text{{{squareordiamond}}} / \sigma_{{\delta(x,y)}}^\text{{{squareordiamond}}}$", labelpad=-2)
        plt.ylabel(rf"Number of {squareordiamond}s")
        plt.margins(y=0.3)
        plt.legend()

      kwargs = {
        "figurekwargs": {"figsize": (6, 6)},
        "verbose": False,
      }

      for samp in "M1_1", "M2_3":
        A = alignmentset(root1=r"\\Bki02\g\heshy", root2=r"\\Bki02\g\heshy\flatw", samp=samp)
        closedlooppulls(A, tagsequence=[4, 2, 6, 8], saveas=os.path.join(here, "squarepull"+samp[1]+".pdf"), plotstyling=functools.partial(plotstyling, squareordiamond="square"), **kwargs)
        closedlooppulls(A, tagsequence=[1, 3, 9, 7], saveas=os.path.join(here, "diamondpull"+samp[1]+".pdf"), plotstyling=functools.partial(plotstyling, squareordiamond="diamond"), **kwargs)

def stitchpulls(*, bki):
  if bki:
    with plt.rc_context(rc=rc):
      def plotstyling(*, fig, ax):
        plt.xlabel(rf"$\delta(x,y)^\text{{overlap}} / \sigma_{{\delta(x,y)}}^\text{{overlap}}$", labelpad=-2)
        plt.ylabel(rf"Number of overlaps")
        plt.margins(y=0.3)
        plt.legend()

      for samp in "M1_1", "M2_3":
        A = alignmentset(root1=r"\\Bki02\g\heshy", root2=r"\\Bki02\g\heshy\flatw", samp=samp)
        for tag in 1, 2, 3, 4:
          plotpairwisealignments(A, tags=[tag], figurekwargs={"figsize": (6, 6)}, stitched=True, pull=True, plotstyling=plotstyling, saveas=os.path.join(here, f"stitch-pull-{tag}-{samp[1]}.pdf"))

def sinewaves(*, bki):
  if bki:
    with plt.rc_context(rc=rc):
      def plotstyling(*, fig, ax, deltaxory, vsxory):
        plt.xlabel(rf"${vsxory}$ (pixels)", labelpad=10)
        plt.ylabel(rf"$\delta {deltaxory}$ (pixels)", labelpad=-5)
        plt.subplots_adjust(bottom=0.15, left=0.18)

      Sample = collections.namedtuple("Sample", "samp root1 root2 name plotsine")
      samples = [
        Sample(root1=r"\\Bki02\g\heshy", root2=r"\\Bki02\g\heshy\flatw", samp="M1_1", name="1", plotsine=lambda tag, **kwargs: kwargs["vsxory"] == {2: "y", 4: "x"}[tag]),
        Sample(root1=r"\\Bki02\g\heshy", root2=r"\\Bki02\g\heshy\flatw", samp="M2_3", name="2", plotsine=lambda tag, **kwargs: kwargs["vsxory"] == {2: "y", 4: "x"}[tag]),
        Sample(root1=r"\\bki02\g\heshy\Clinical_Specimen_BMS_03", root2=r"\\Bki02\g\flatw", samp="TS19_0181_A_1_3_BMS_MITRE", name="AKY", plotsine=lambda tag, **kwargs: tag==4 and kwargs["deltaxory"] == kwargs["vsxory"] == "x"),
      ]

      for samp, root1, root2, name, plotsine in samples:
        A = alignmentset(root1=root1, root2=root2, samp=samp)
        kwargs = {}
        for kwargs["deltaxory"] in "xy":
          for kwargs["vsxory"] in "xy":
            for tag in 2, 4:
              alignmentshiftprofile(A, tag=tag, plotsine=plotsine(tag, **kwargs), sinetext=True, figurekwargs={"figsize": (6, 6)}, plotstyling=functools.partial(plotstyling, **kwargs), saveas=os.path.join(here, f"sine-wave-{tag}-{kwargs['deltaxory']}{kwargs['vsxory']}-{name}.pdf"), **kwargs)

if __name__ == "__main__":
  class EqualsEverything:
    def __eq__(self, other): return True
  p = argparse.ArgumentParser()
  p.add_argument("--bki", action="store_true")
  p.add_argument("--units", choices=("fast", "safe"), default="safe")
  g = p.add_mutually_exclusive_group()
  g.add_argument("--all", action="store_const", dest="which", const=EqualsEverything(), default=EqualsEverything())
  g.add_argument("--maximize", action="store_const", dest="which", const="maximize")
  g.add_argument("--overlap", action="store_const", dest="which", const="overlap")
  g.add_argument("--xcorrelation", "--cross-correlation", action="store_const", dest="which", const="xcorrelation")
  g.add_argument("--islands", action="store_const", dest="which", const="islands")
  g.add_argument("--alignmentresults", action="store_const", dest="which", const="alignmentresults")
  g.add_argument("--scanning", action="store_const", dest="which", const="scanning")
  g.add_argument("--squarepulls", action="store_const", dest="which", const="squarepulls")
  g.add_argument("--stitchpulls", action="store_const", dest="which", const="stitchpulls")
  g.add_argument("--sinewaves", action="store_const", dest="which", const="sinewaves")
  args = p.parse_args()

  units.setup(args.units)

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
  if args.which == "squarepulls":
    squarepulls(bki=args.bki)
  if args.which == "stitchpulls":
    stitchpulls(bki=args.bki)
  if args.which == "sinewaves":
    sinewaves(bki=args.bki)
