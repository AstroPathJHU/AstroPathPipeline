import abc, argparse, numpy as np, pathlib

from ..baseclasses.sample import ReadRectangles
from ..utilities import units
from .dustspeck import DustSpeckFinder

class BadRegionFinderSample(ReadRectangles):
  @abc.abstractmethod
  def makebadregionfinder(self, *args, **kwargs): pass

  @property
  def layer(self): return 1

  def run(self, *, plotsdir=None, **kwargs):
    rawimages = self.getrawlayers("flatWarpDAPI")
    result = np.empty(shape=rawimages.shape, dtype=bool)

    if plotsdir is not None:
      plotsdir = pathlib.Path(plotsdir)
      plotsdir.mkdir(exist_ok=True)
      showkwargs = {}
      for name in "scale":
        if name in kwargs:
          showkwargs[name] = kwargs.pop(name)

    nbad = 0
    for i, (r, rawimage) in enumerate(zip(self.rectangles, rawimages)):
      self.logger.info(f"looking for bad regions in HPF {i+1}/{len(self.rectangles)}")
      f = self.makebadregionfinder(rawimage, logger=self.logger)
      result[i] = f.badregions(**kwargs)
      if np.any(result[i]):
        nbad += 1
        if plotsdir is not None:
          f.show(saveas=plotsdir/f"{r.n}.pdf", alpha=0.6, **kwargs, **showkwargs)
          f.show(saveas=plotsdir/f"{r.n}_image.pdf", alpha=0, **kwargs, **showkwargs)

    logfunction = self.logger.warningglobal if nbad > 0 else self.logger.info
    logfunction(f"{nbad} HPFs had bad regions")
    return result

class DustSpeckFinderSample(BadRegionFinderSample):
  def makebadregionfinder(self, *args, **kwargs):
    return DustSpeckFinder(*args, **kwargs)

  @property
  def logmodule(self): return "dustspeckfinder"

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  g = p.add_mutually_exclusive_group(required=True)
  g.add_argument("--dust-speck", dest="cls", action="store_const", const=DustSpeckFinderSample)
  p.add_argument("root1")
  p.add_argument("root2")
  p.add_argument("samp")
  p.add_argument("--units", choices=("fast", "safe"), default="fast")
  p.add_argument("--plotsdir", type=pathlib.Path)
  args = p.parse_args()

  units.setup(args.units)

  result = args.cls(args.root1, args.root2, args.samp).run(plotsdir=args.plotsdir)
