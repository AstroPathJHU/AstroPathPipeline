import abc, argparse, pathlib

from ..baseclasses.cohort import FlatwCohort
from .sample import DustSpeckFinderSample

class BadRegionFinderCohort(FlatwCohort):
  def __init__(self, *args, uselogfiles=False, **kwargs):
    return super().__init__(*args, uselogfiles=uselogfiles, **kwargs)

  def runsample(self, sample, *, plotsdir=None, **kwargs):
    if plotsdir is not None:
      plotsdir = pathlib.Path(plotsdir)
      plotsdir.mkdir(exist_ok=True)
      kwargs["plotsdir"] = plotsdir/sample.SlideID
    sample.run(**kwargs)

  def initiatesample(self, samp, **kwargs):
    return self.badregionfindersampleclass(self.root1, self.root2, samp, uselogfiles=self.uselogfiles, **kwargs)

  @abc.abstractproperty
  def badregionfindersampleclass(self): pass

class DustSpeckFinderCohort(BadRegionFinderCohort):
  badregionfindersampleclass = DustSpeckFinderSample
  logmodule = badregionfindersampleclass.logmodule

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  g = p.add_mutually_exclusive_group(required=True)
  g.add_argument("--dust-speck", dest="cls", action="store_const", const=DustSpeckFinderCohort)
  p.add_argument("root1")
  p.add_argument("root2")
  p.add_argument("--plotsdir", type=pathlib.Path)
  args = p.parse_args()

  args.cls(args.root1, args.root2).run(plotsdir=args.plotsdir)
