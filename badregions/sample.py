import abc, argparse, numpy as np, pathlib

from ..baseclasses.sample import ReadRectangles
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
          showkwargs[name] = kwargs.pop("name")

    for i, (r, rawimage) in enumerate(zip(self.rectangles, rawimages)):
      f = self.makebadregionfinder(rawimage)
      result[i] = f.badregions(**kwargs)
      if plotsdir is not None and np.any(result[i]):
        f.show(saveas=plotsdir/f"{r.n}.pdf", alpha=0.6, **kwargs, **showkwargs)
        f.show(saveas=plotsdir/f"{r.n}_image.pdf", alpha=0, **kwargs, **showkwargs)

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
  p.add_argument("--plotsdir", type=pathlib.Path)
  args = p.parse_args()

  result = args.cls(args.root1, args.root2, args.samp).run(plotsdir=args.plotsdir)
  print(f"{np.count_nonzero(np.any(result, axis=(1, 2)))} HPFs had bad regions")
