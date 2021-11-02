import argparse, numpy as np, pathlib, PIL, shutil, tempfile
from astropath.slides.annowarp.annowarpsample import AnnoWarpSampleInformTissueMask
from astropath.slides.annowarp.visualization import showannotation
from astropath.shared.csvclasses import Region
from astropath.utilities import units

here = pathlib.Path(__file__).parent
data = here/".."/".."/"test"/"data"
zoomroot = here/".."/".."/"test"/"data"/"reference"/"zoom"
annowarproot = here/".."/".."/"test"/"data"/"reference"/"annowarp"
alignroot = here/".."/".."/"test"/"data"/"reference"/"alignment"/"component_tiff"
prepdbroot = here/".."/".."/"test"/"data"/"reference"/"prepdb"
samp = "M206"

def makeplots():
  with tempfile.TemporaryDirectory() as dbloadroot:
    dbloadroot = pathlib.Path(dbloadroot)
    (dbloadroot/samp/"dbload").mkdir(parents=True)
    for root in prepdbroot, annowarproot, alignroot:
      for folder in root/samp, root/samp/"dbload":
        for filename in folder.glob("*.csv"):
          shutil.copy(filename, dbloadroot/samp/"dbload")
    from ...test.testzoom import gunzipreference
    gunzipreference(samp)
    with AnnoWarpSampleInformTissueMask(data, samp, zoomroot=zoomroot, dbloadroot=dbloadroot, annotationsynonyms={"Good tisue": "Good tissue"}) as A:
      #annotationsynonyms is in case of an unlucky collision in the jenkins tests
      #where this runs at the same time as M206 in prepdb
      warpedregions = A.readtable(A.regionscsv, Region)

      with A.using_images() as (wsi, fqptiff):
        zoomlevel = fqptiff.zoomlevels[0]
        qptiff = PIL.Image.fromarray(zoomlevel[A.qptifflayer-1].asarray())
        wsi = np.asarray(wsi)

      oneqppixel = units.onepixel(A.apscale)
      ylim = np.array((9500, 10500)) * oneqppixel
      xlim = np.array((7000, 7500)) * oneqppixel
      if tuple(xlim):
        xlimpscale = units.convertpscale(xlim, A.apscale, A.pscale)
      else:
        xlimpscale = xlim
      if tuple(ylim):
        ylimpscale = units.convertpscale(ylim, A.apscale, A.pscale)
      else:
        ylimpscale = ylim

      showannotation(qptiff, A.regions, vertices=A.apvertices, imagescale=A.apscale, figurekwargs={}, ylim=ylim, xlim=xlim, saveas=here/"qptiff.pdf")
      showannotation(wsi, warpedregions, imagescale=A.pscale, figurekwargs={}, ylim=ylimpscale, xlim=xlimpscale, saveas=here/"wsi.pdf")

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--units", choices=("safe", "fast"), default="safe")
  args = p.parse_args()

  units.setup(args.units)

  makeplots()
