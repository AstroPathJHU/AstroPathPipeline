#!/usr/bin/env python

import argparse, functools, matplotlib.pyplot as plt, pathlib
from astropathcalibration.badregions.dustspeck import DustSpeckFinder
from astropathcalibration.badregions.sample import DustSpeckFinderSample
from astropathcalibration.utilities import units

here = pathlib.Path(__file__).parent
data = here/".."/".."/"test"/"data"
interactive = False

rc = {
  "text.usetex": True,
  "text.latex.preamble": [r"\usepackage{amsmath}\usepackage{siunitx}"],
  "font.size": 20,
  "figure.subplot.bottom": 0.12,
}

@functools.lru_cache()
def getimage(which, i):
  if which == "dust" and i == 0:
    A = DustSpeckFinderSample(data, data/"flatw", "M55_1", selectrectangles=[678])
    return A.rectangles[0].image

def dust(*, remake):
  save = {0: here/"dust.pdf", 0.6: here/"dustdetected.pdf"}
  if not remake and all(_.exists() for _ in save.values()): return

  def plotstyling(fig, ax):
    plt.xticks([])
    plt.yticks([])

  with plt.rc_context(rc):
    f = DustSpeckFinder(getimage("dust", 0))

    for alpha, saveas in save.items():
      f.show(alpha=alpha, saveas=saveas, plotstyling=plotstyling, scale=0.2)

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
  g.add_argument("--dust", action="store_const", dest="which", const="dust")
  args = p.parse_args()

  units.setup(args.units)
  interactive = args.interactive

  if args.which == "dust":
    dust(remake=args.remake)
