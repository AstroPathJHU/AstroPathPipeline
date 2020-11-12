import argparse, pathlib, re

from ..baseclasses.cohort import DbloadCohort, FlatwCohort
from ..utilities import units
from .sample import DustSpeckFinderSample

class BadRegionFinderCohort(DbloadCohort, FlatwCohort):
  def __init__(self, *args, uselogfiles=True, **kwargs):
    return super().__init__(*args, uselogfiles=uselogfiles, **kwargs)

  def runsample(self, sample, *, plotsdir=None, **kwargs):
    if plotsdir is not None:
      plotsdir = pathlib.Path(plotsdir)
      plotsdir.mkdir(exist_ok=True)
      kwargs["plotsdir"] = plotsdir/sample.SlideID
    sample.run(**kwargs)

class DustSpeckFinderCohort(BadRegionFinderCohort):
  sampleclass = DustSpeckFinderSample
  logmodule = sampleclass.logmodule

def main(args=None):
  p = argparse.ArgumentParser()
  g = p.add_mutually_exclusive_group(required=True)
  g.add_argument("--dust-speck", dest="cls", action="store_const", const=DustSpeckFinderCohort)
  p.add_argument("root1")
  p.add_argument("root2")
  p.add_argument("--units", choices=("fast", "safe"), default="fast")
  p.add_argument("--plotsdir", type=pathlib.Path)
  g = p.add_mutually_exclusive_group()
  g.add_argument("--sampleregex", type=re.compile)
  g.add_argument("--skip-existing-plots", action="store_true")
  p.add_argument("--debug", action="store_true")
  args = p.parse_args(args=args)

  units.setup(args.units)

  kwargs = {"root": args.root1, "root2": args.root2, "debug": args.debug}
  if args.sampleregex is not None:
    kwargs["filter"] = lambda sample: args.sampleregex.match(sample.SlideID)
  elif args.skip_existing_plots:
    kwargs["filter"] = lambda sample: not (args.plotsdir/sample.SlideID).exists()

  args.cls(**kwargs).run(plotsdir=args.plotsdir)

if __name__ == "__main__":
  main()
