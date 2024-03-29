#!/usr/bin/env python

import argparse, collections, contextlib, functools, os, matplotlib.patches as patches, matplotlib.pyplot as plt, numpy as np, pathlib, scipy.interpolate
from astropath.slides.align.plots import shiftplotprofile, closedlooppulls, plotpairwisealignments, shiftplot2D
from astropath.slides.align.isotropy import isotropy, stitchingisotropy
from astropath.slides.align.alignsample import AlignSample
from astropath.utilities import units

here = pathlib.Path(__file__).parent
data = here/".."/".."/"test"/"data"
interactive = False

rc = {
  "text.usetex": True,
  "text.latex.preamble": r"\usepackage{amsmath}\usepackage{siunitx}",
  "font.size": 20,
  "figure.subplot.bottom": 0.12,
}

stack = contextlib.ExitStack()

@functools.lru_cache()
def __alignsample(*, root1, shardedim3root, samp, dapi, dbloadroot, **kwargs):
  if dapi:
    A = alignsample(root1=root1, shardedim3root=shardedim3root, samp=samp, dbloadroot=dbloadroot, **kwargs)
    A.getDAPI()
    return A

  if root1 is shardedim3root is samp is dbloadroot is None:
    return alignsample(samp="M21_1", **kwargs)

  if root1 is shardedim3root is dbloadroot is None:
    if samp == "M21_1": root1, shardedim3root = data, data/"flatw"
    elif samp == "M1_1" or samp == "M2_3": root1, shardedim3root = r"\\Bki02\g\heshy", r"\\Bki02\g\heshy\flatw"
    elif samp == "TS19_0181_A_1_3_BMS_MITRE": root1, shardedim3root = r"\\bki02\g\heshy\Clinical_Specimen_BMS_03", r"\\Bki02\g\flatw"
    elif samp == "L1_4": root1, shardedim3root = r"\\bki04\Clinical_Specimen_2", r"\\bki02\g\heshy\Clinical_Specimen_2"
    elif samp == "PZ1": root1, shardedim3root = r"\\bki03\Clinical_Specimen_4", r"\\bki02\g\heshy\Clinical_Specimen_4"
    elif samp == "ML1603474_BMS069_5_21" or samp == "ML1603480_BMS078_5_22": root1, shardedim3root = r"\\bki03\Clinical_Specimen_BMS_01", r"\\bki02\g\heshy\Clinical_Specimen_BMS_01"
    elif samp == "M115": root1, shardedim3root = r"\\bki07\Clinical_Specimen_11", r"\\bki07\flatw_11"
    else: raise ValueError(samp)
    return alignsample(root1=root1, shardedim3root=shardedim3root, dbloadroot=dbloadroot, samp=samp, **kwargs)

  if dbloadroot is None:
    return alignsample(root1=root1, shardedim3root=shardedim3root, samp=samp, dbloadroot=root1, **kwargs)

  A = stack.enter_context(AlignSample(root1, shardedim3root, dbloadroot=dbloadroot, samp=samp, interactive=interactive, **kwargs))

  A.readalignments()
  A.readstitchresult()
  return A

def alignsample(*, root1=None, shardedim3root=None, samp=None, dapi=False, dbloadroot=None, **kwargs):
  return __alignsample(root1=root1, shardedim3root=shardedim3root, samp=samp, dapi=dapi, dbloadroot=dbloadroot, **kwargs)

def overlap():
  A = alignsample(dapi=True)
  o = A.overlaps[140]
  with plt.rc_context(rc=rc):
    o.showimages(shifted=False, normalize=1000, ticks=True, saveas=here/"overlap-notshifted.pdf")
    o.showimages(shifted=True, normalize=1000, ticks=True, saveas=here/"overlap-shifted.pdf")

def xcorrelation():
  A = alignsample(dapi=True)
  o = A.overlaps[140]
  with plt.rc_context(rc=rc):
    o.align(savebigimage=here/"overlap-xcorrelation.pdf", alreadyalignedstrategy="overwrite", debug=True)

  o = A.overlaps[203]
  with plt.rc_context(rc=rc):
    o.showimages(shifted=False, normalize=100, ticks=True, saveas=here/"overlap-bad.pdf")
    o.align(savebigimage=here/"overlap-xcorrelation-bad.pdf", alreadyalignedstrategy="overwrite", debug=True)

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
    plt.savefig(here/"1Dmaximization.pdf")

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
    plt.savefig(here/"1Dmaximizationwitherror.pdf")

    plt.close(fig)

def islands():
  A = alignsample()
  with plt.rc_context(rc=rc):
    plt.imshow(A.qptiffjpgimage())
    plt.xticks([])
    plt.yticks([])
    plt.savefig(here/"islands.pdf")
    plt.close()

def alignmentresults(*, bki, remake):
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
    for samp, name in (
      ("M21_1", "vectra"),
      ("M1_1", "vectra-big"),
      ("TS19_0181_A_1_3_BMS_MITRE", "AKY"),
      ("PZ1", "JHUPolaris"),
      ("ML1603480_BMS078_5_22", "BMS"),
    ):
      if samp != "M21_1" and not bki: continue
      for tag in 1, 2, 3, 4:
        filename1, filename2 = here/f"alignment-result-{name}-{tag}.pdf", here/f"stitch-result-{name}-{tag}.pdf"
        if samp != "M21_1" and filename1.exists() and filename2.exists() and not remake: continue
        A = alignsample(samp=samp)
        errorbars = samp == "M21_1"
        plotpairwisealignments(A, tags=[tag], saveas=filename1, errorbars=errorbars, **kwargs)
        plotpairwisealignments(A, tags=[tag], stitched=True, saveas=filename2, errorbars=errorbars, **kwargs)

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
    plt.savefig(here/"scanning.pdf")
    plt.close()

def squarepulls(*, bki, testing, remake):
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

    plt.savefig(here/"squarepulldiagram.pdf")

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

    plt.savefig(here/"diamondpulldiagram.pdf")

    plt.close()

    if bki or testing:

      def plotstyling(*, fig, ax, squareordiamond):
        plt.xlabel(rf"$\delta(x,y)^\text{{{squareordiamond}}} / \sigma_{{\delta(x,y)}}^\text{{{squareordiamond}}}$", labelpad=-2)
        plt.ylabel(rf"Number of {squareordiamond}s")
        plt.margins(y=0.3)
        plt.legend()

      kwargs = {
        "figurekwargs": {"figsize": (6, 6)},
        "verbose": False,
      }

      samples = ("M1_1", "M2_3") if bki else (None,)

      for samp in samples:
        plotid = samp[1] if samp else "_test"
        saveas = here/("squarepull"+plotid+".pdf")
        if remake or not os.path.exists(saveas):
          A = alignsample(samp=samp)
          closedlooppulls(A, tagsequence=[4, 2, 6, 8], saveas=saveas, plotstyling=functools.partial(plotstyling, squareordiamond="square"), **kwargs)
        saveas = here/("diamondpull"+plotid+".pdf")
        if remake or not os.path.exists(saveas):
          A = alignsample(samp=samp)
          closedlooppulls(A, tagsequence=[1, 3, 9, 7], saveas=here/("diamondpull"+plotid+".pdf"), plotstyling=functools.partial(plotstyling, squareordiamond="diamond"), **kwargs)

def stitchpulls(*, bki, testing, remake):
  if bki or testing:
    with plt.rc_context(rc=rc):
      def plotstyling(*, fig, ax):
        plt.xlabel(r"$\delta(x,y)^\text{overlap} / \sigma_{\delta(x,y)}^\text{overlap}$", labelpad=-2)
        plt.ylabel(r"Number of overlaps")
        plt.margins(y=0.3)
        plt.legend()

      samples = ("M1_1", "M2_3") if bki else (None,)

      for samp in samples:
        plotid = samp[1] if samp else "test"
        for tag in 1, 2, 3, 4:
          saveas=os.path.join(here, f"stitch-pull-{tag}-{plotid}.pdf")
          if os.path.exists(saveas) and not remake: continue
          A = alignsample(samp=samp)
          plotpairwisealignments(
            A,
            tags=[tag],
            figurekwargs={"figsize": (6, 6)},
            stitched=True,
            pull=True,
            plotstyling=plotstyling,
            saveas=saveas
          )

def sinewaves(*, bki, testing, remake):
  if bki or testing:
    with plt.rc_context(rc=rc):
      def plotstyling(*, fig, ax, deltaxory, vsxory):
        ax.set_xlabel(rf"${vsxory}$ (pixels)", labelpad=10)
        ax.set_ylabel(rf"$\delta {deltaxory}$ (pixels)", labelpad=0)
        fig.subplots_adjust(bottom=0.15, left=0.21)
        ymin, ymax = ax.get_ylim()
        ymin = min(ymin, -ymax, -10)
        ymax = -ymin
        ax.set_ylim(ymin, ymax)

      class Sample(collections.namedtuple("Sample", "samp name plotsine sinetext guessparameters")):
        def __new__(cls, *, plotsine=lambda **kwargs: True, sinetext=lambda **kwargs: True, guessparameters=lambda **kwargs: None, **kwargs):
          return super().__new__(cls, plotsine=plotsine, sinetext=sinetext, guessparameters=guessparameters, **kwargs)

      samples = [
        Sample(samp="M1_1", name="1"),
        Sample(samp="M2_3", name="2"),
        Sample(samp="TS19_0181_A_1_3_BMS_MITRE", name="AKY", plotsine=lambda deltaxory, vsxory, **kwargs: deltaxory == vsxory == "x"),
        Sample(samp="PZ1", name="JHUPolaris"),
        Sample(samp="ML1603480_BMS078_5_22", name="BMS", plotsine=lambda deltaxory, vsxory, **kwargs: deltaxory == vsxory == "x"),
      ] if bki else [
        Sample(samp=None, name="test"),
      ]

      for samp, name, plotsine, sinetext, guessparameters in samples:
        alignsamplekwargs = {"samp": samp}
        alignsamplekwargs = {k: v for k, v in alignsamplekwargs.items() if v is not None}
        kwargs = {}
        for kwargs["deltaxory"] in "xy":
          for kwargs["vsxory"] in "xy":
            if kwargs["deltaxory"] != kwargs["vsxory"]: continue
            saveas = os.path.join(here, f"sine-wave-{kwargs['deltaxory']}{kwargs['vsxory']}-{name}.pdf")
            if os.path.exists(saveas) and not remake: continue
            A = alignsample(**alignsamplekwargs)
            shiftplotprofile(
              A,
              plotsine=plotsine(**kwargs),
              sinetext=sinetext(**kwargs),
              guessparameters=guessparameters(**kwargs),
              figurekwargs={"figsize": (6, 6)},
              plotstyling=functools.partial(plotstyling, **kwargs),
              saveas=saveas,
              **kwargs
            )

def plots2D(*, bki, testing, remake):
  if bki or testing:
    with plt.rc_context(rc=rc):
      def plotstyling(*, fig, ax, cbar, xory, subplotkwargs):
        ax.set_xlabel(r"$x$ (pixels)", labelpad=10)
        ax.set_ylabel(r"$y$ (pixels)", labelpad=0)
        cbar.set_label(f"$\delta {xory}$ (pixels)")
        fig.subplots_adjust(**subplotkwargs)

      class Sample(collections.namedtuple("Sample", "samp name figurekwargs subplotkwargs")):
        def __new__(cls, *, figurekwargs={}, subplotkwargs={}, **kwargs):
          return super().__new__(cls, figurekwargs=figurekwargs, subplotkwargs=subplotkwargs, **kwargs)

      samples = [
        Sample(samp="M1_1", name="JHUVectra", figurekwargs={"figsize": (6, 5)}, subplotkwargs={"bottom": 0.15, "left": 0.17, "right": 0.87}),
        Sample(samp="TS19_0181_A_1_3_BMS_MITRE", name="AKY", figurekwargs={"figsize": (5.7, 6)}, subplotkwargs={"bottom": 0.15, "left": 0.16, "right": 0.86}),
        Sample(samp="PZ1", name="JHUPolaris", figurekwargs={"figsize": (6.7, 6)}, subplotkwargs={"bottom": 0.15, "left": 0.16, "right": 0.86}),
        Sample(samp="ML1603480_BMS078_5_22", name="BMS", figurekwargs={"figsize": (5.7, 6)}, subplotkwargs={"bottom": 0.15, "left": 0.16, "right": 0.86}),
        Sample(samp="ML1603474_BMS069_5_21", name="BMS2", figurekwargs={"figsize": (5.7, 6)}, subplotkwargs={"bottom": 0.15, "left": 0.16, "right": 0.86}),
      ] if bki else [
        Sample(samp=None, name="test", figurekwargs={"figsize": (6, 4)}, subplotkwargs={"bottom": 0.15, "left": 0.16, "right": 0.86}),
      ]

      for samp, name, figurekwargs, subplotkwargs in samples:
        alignsamplekwargs = {"samp": samp}
        alignsamplekwargs = {k: v for k, v in alignsamplekwargs.items() if v is not None}
        saveasx, saveasy = (here/f"2D-shifts-{name}-{xy}.pdf" for xy in "xy")
        if saveasx.exists() and saveasy.exists() and not remake: continue
        A = alignsample(**alignsamplekwargs)
        shiftplot2D(
          A,
          figurekwargs=figurekwargs,
          saveasx=saveasx,
          saveasy=saveasy,
          plotstyling=functools.partial(plotstyling, subplotkwargs=subplotkwargs),
        )

def isotropyhist(*, bki, testing, remake):
  if bki or testing:
    with plt.rc_context(rc=rc):
      def plotstyling(fig, ax):
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"number of overlaps")
        low, hi = ax.get_ylim()
        ax.set_ylim(top = hi + (hi-low)*.4)
        plt.legend(title="Fourier series up to order", ncol=3)

      class Sample(collections.namedtuple("Sample", "samp name")):
        def __new__(cls, **kwargs):
          return super().__new__(cls, **kwargs)

      samples = [
        Sample(samp="M1_1", name="JHUVectra"),
        Sample(samp="TS19_0181_A_1_3_BMS_MITRE", name="AKY"),
        Sample(samp="PZ1", name="JHUPolaris"),
        Sample(samp="ML1603480_BMS078_5_22", name="BMS"),
      ] if bki else [
        Sample(samp=None, name="test"),
      ]
      figurekwargs = {"figsize": (6, 6)}

      for samp, name in samples:
        try:
          currentstitchresultisfor = "all"
          for overlaps in "all", "edges", "corners", "bestRMS":
            for tag in 1, 2, 3, 4:
              alignsamplekwargs = {"samp": samp}
              alignsamplekwargs = {k: v for k, v in alignsamplekwargs.items() if v is not None}
              saveas = here/f"isotropy-histogram-{name}-{tag}-{overlaps}.pdf"
              if saveas.exists() and not remake: continue
              A = alignsample(**alignsamplekwargs)
              A.logger.info("%s %s", tag, overlaps)
              if currentstitchresultisfor != overlaps:
                if currentstitchresultisfor == "all": oldstitchresult = A.stitchresult
                currentstitchresultisfor = overlaps
                if overlaps == "edges":
                  A.applystitchresult(A.stitch(scalecorners=0, saveresult=False))
                elif overlaps == "corners":
                  A.applystitchresult(A.stitch(scaleedges=0, saveresult=False))
                elif overlaps == "bestRMS":
                  A.applystitchresult(A.stitch(scaleedges=.2, scalecorners=.8, saveresult=False))
                else:
                  raise ValueError(overlaps)
              isotropy(
                A,
                tags=[tag],
                stitched=True,
                plotstyling=plotstyling,
                figurekwargs=figurekwargs,
                saveas=saveas,
              )
        finally:
          if currentstitchresultisfor != "all":
            A.applystitchresult(oldstitchresult)

def isotropyscatter(*, bki, testing, remake):
  if bki or testing:
    with plt.rc_context(rc=rc):
      def plotstyling(fig, ax, isotropyorRMS):
        ax.set_xlabel(r"$\alpha$")
        if isotropyorRMS == "isotropy":
          ax.set_ylabel(r"Fourier amplitude")
          low, hi = ax.get_ylim()
          ax.set_ylim(top = hi + (hi-low)*.4)
          plt.legend(ncol=3)
        elif isotropyorRMS == "rms":
          ax.set_ylabel(r"RMS $\delta\vec{r}$ (pixels)")

      class Sample(collections.namedtuple("Sample", "samp name")):
        def __new__(cls, **kwargs):
          return super().__new__(cls, **kwargs)

      samples = [
        Sample(samp="M1_1", name="JHUVectra"),
        Sample(samp="TS19_0181_A_1_3_BMS_MITRE", name="AKY"),
        Sample(samp="PZ1", name="JHUPolaris"),
        Sample(samp="ML1603480_BMS078_5_22", name="BMS"),
      ] if bki else [
        Sample(samp=None, name="test"),
      ]
      figurekwargs = {"figsize": (6, 6)}

      for samp, name in samples:
        alignsamplekwargs = {"samp": samp}
        alignsamplekwargs = {k: v for k, v in alignsamplekwargs.items() if v is not None}
        saveas = here/f"stitching-{{isotropyorRMS}}-{name}-{{tag}}.pdf"
        if not remake and all(
          pathlib.Path(str(saveas).format(isotropyorRMS=thing, tag=tag)).exists()
          for tag in (1, 2, 3, 4, "all")
          for thing in ("isotropy", "rms")
          if not (thing == "isotropy" and tag == "all")
        ): continue
        A = alignsample(**alignsamplekwargs)
        oldstitchresult = A.stitchresult
        try:
          stitchingisotropy(
            A,
            plotstyling=plotstyling,
            figurekwargs=figurekwargs,
            saveas=saveas,
          )
        finally:
          A.applystitchresult(oldstitchresult)

if __name__ == "__main__":
  class EqualsEverything:
    def __eq__(self, other): return True
  p = argparse.ArgumentParser()
  g = p.add_mutually_exclusive_group()
  g.add_argument("--bki", action="store_true")
  g.add_argument("--testing", action="store_true")
  p.add_argument("--remake", action="store_true")
  p.add_argument("--interactive", action="store_true")
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
  g.add_argument("--2dplots", action="store_const", dest="which", const="2dplots")
  g.add_argument("--isotropy-histograms", action="store_const", dest="which", const="isotropyhist")
  g.add_argument("--isotropy-scatter", action="store_const", dest="which", const="isotropyscatter")
  args = p.parse_args()

  units.setup(args.units)
  interactive = args.interactive

  if args.which == "maximize":
    maximize1D()
  if args.which == "overlap":
    overlap()
  if args.which == "xcorrelation":
    xcorrelation()
  if args.which == "islands":
    islands()
  if args.which == "alignmentresults":
    alignmentresults(bki=args.bki, remake=args.remake)
  if args.which == "scanning":
    scanning()
  if args.which == "squarepulls":
    squarepulls(bki=args.bki, testing=args.testing, remake=args.remake)
  if args.which == "stitchpulls":
    stitchpulls(bki=args.bki, testing=args.testing, remake=args.remake)
  if args.which == "sinewaves":
    sinewaves(bki=args.bki, testing=args.testing, remake=args.remake)
  if args.which == "2dplots":
    plots2D(bki=args.bki, testing=args.testing, remake=args.remake)
  if args.which == "isotropyhist":
    isotropyhist(bki=args.bki, testing=args.testing, remake=args.remake)
  if args.which == "isotropyscatter":
    isotropyscatter(bki=args.bki, testing=args.testing, remake=args.remake)
