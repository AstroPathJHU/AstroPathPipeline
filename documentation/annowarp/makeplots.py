import argparse, numpy as np, pathlib, PIL
from astropath_calibration.annowarp.annowarpsample import AnnoWarpSample
from astropath_calibration.annowarp.visualization import showannotation
from astropath_calibration.baseclasses.csvclasses import Region
from astropath_calibration.utilities import units
from astropath_calibration.utilities.tableio import readtable

here = pathlib.Path(__file__).parent
data = here/".."/".."/"test"/"data"
zoomroot = here/".."/".."/"test"/"reference"/"zoom"
annowarproot = here/".."/".."/"test"/"annowarp_test_for_jenkins"
samp = "M206"

def makeplots():
  from ...test.testzoom import gunzipreference
  gunzipreference(samp)
  A = AnnoWarpSample(data, samp, zoomroot=zoomroot)
  warpedregions = readtable(annowarproot/samp/A.newregionscsv.name, Region, extrakwargs={"apscale": A.apscale, "pscale": A.pscale})

  with A.using_images() as (wsi, fqptiff):
    zoomlevel = fqptiff.zoomlevels[0]
    apscale = zoomlevel.qpscale
    qptiff = PIL.Image.fromarray(zoomlevel[A.qptifflayer-1].asarray())
    wsi = np.asarray(wsi)

  oneqppixel = units.onepixel(apscale)
  ylim = np.array((9500, 10500)) * oneqppixel
  xlim = np.array((7000, 7500)) * oneqppixel
  if tuple(xlim):
    xlimpscale = units.convertpscale(xlim, apscale, A.pscale)
  else:
    xlimpscale = xlim
  if tuple(ylim):
    ylimpscale = units.convertpscale(ylim, apscale, A.pscale)
  else:
    ylimpscale = ylim

  showannotation(qptiff, A.regions, vertices=A.vertices, qpscale=A.apscale, imagescale=apscale, figurekwargs={}, ylim=ylim, xlim=xlim, saveas=here/"qptiff.pdf")
  showannotation(wsi, warpedregions, qpscale=apscale, imagescale=A.pscale, figurekwargs={}, ylim=ylimpscale, xlim=xlimpscale, saveas=here/"wsi.pdf")

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--units", choices=("safe", "fast"), default="safe")
  args = p.parse_args()

  units.setup(args.units)

  makeplots()
