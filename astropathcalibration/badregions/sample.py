import abc, argparse, numpy as np, pathlib

from ..baseclasses.sample import ReadRectanglesIm3
from ..utilities import units
from .dustspeck import DustSpeckFinder
from .tissuefold import TissueFoldFinderSimple

class BadRegionFinderSample(ReadRectanglesIm3):
  def __init__(self, *args, filetype="flatWarp", **kwargs):
    super().__init__(*args, filetype=filetype, **kwargs)
  @abc.abstractmethod
  def makebadregionfinder(self, *args, **kwargs): pass

  def run(self, *, plotsdir=None, show=False, **kwargs):
    result = np.empty(shape=(len(self.rectangles), *self.rectangles[0].imageshape), dtype=bool)

    if plotsdir is not None:
      plotsdir = pathlib.Path(plotsdir)
      plotsdir.mkdir(exist_ok=True)

    showkwargs = {}
    for name in "scale":
      if name in kwargs:
        showkwargs[name] = kwargs.pop(name)

    nbad = 0
    for i, r in enumerate(self.rectangles):
      self.logger.info(f"loading image for HPF {i+1}/{len(self.rectangles)}")
      with r.using_image() as image:
        self.logger.info("looking for bad regions")
        f = self.makebadregionfinder(image, logger=self.logger)
        result[i] = f.badregions(**kwargs)
        if np.any(result[i]):
          nbad += 1
          if plotsdir is not None:
            f.show(saveas=plotsdir/f"{r.n}.pdf", alpha=0.6, **kwargs, **showkwargs)
            f.show(saveas=plotsdir/f"{r.n}_image.pdf", alpha=0, **kwargs, **showkwargs)
          if show:
            f.show(alpha=0.6, **kwargs, **showkwargs)

    logfunction = self.logger.warningglobal if nbad > 0 else self.logger.info
    logfunction(f"{nbad} HPFs had bad regions")
    return result

class DustSpeckFinderSample(BadRegionFinderSample):
  multilayer = True
  def makebadregionfinder(self, *args, **kwargs):
    return DustSpeckFinder(*args, **kwargs)

  @property
  def logmodule(self): return "dustspeckfinder"

class TissueFoldFinderSample(BadRegionFinderSample):
  def makebadregionfinder(self, *args, **kwargs):
    return TissueFoldFinderSimple(*args, **kwargs)

  @property
  def logmodule(self): return "tissuefoldfinder"

def main(args=None):
  p = argparse.ArgumentParser()
  g = p.add_mutually_exclusive_group(required=True)
  g.add_argument("--dust-speck", dest="cls", action="store_const", const=DustSpeckFinderSample)
  p.add_argument("root1")
  p.add_argument("root2")
  p.add_argument("samp")
  p.add_argument("--units", choices=("fast", "safe"), default="fast")
  p.add_argument("--plotsdir", type=pathlib.Path)
  args = p.parse_args(args=args)

  units.setup(args.units)

  result = args.cls(args.root1, args.root2, args.samp).run(plotsdir=args.plotsdir)

if __name__ == "__main__":
  main()
