import argparse, pathlib, re

from ..baseclasses.cohort import DbloadCohort, FlatwCohort
from ..utilities import units
from .zoom import Zoom

class ZoomCohort(DbloadCohort, FlatwCohort):
  def __init__(self, *args, zoomroot, **kwargs):
    self.__zoomroot = zoomroot
    super().__init__(*args, **kwargs)

  sampleclass = Zoom
  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "zoomroot": self.__zoomroot}

  def runsample(self, sample):
    #sample.logger.info(f"{sample.ntiles} {len(sample.rectangles)}")
    return sample.zoom()

  @property
  def logmodule(self): return "zoom"

def main(args=None):
  p = argparse.ArgumentParser()
  p.add_argument("root1", type=pathlib.Path)
  p.add_argument("root2", type=pathlib.Path)
  p.add_argument("--debug", action="store_true")
  g = p.add_mutually_exclusive_group()
  g.add_argument("--sampleregex", type=re.compile)
  g.add_argument("--skip-aligned", action="store_true")
  p.add_argument("--units", choices=("safe", "fast"), default="fast")
  p.add_argument("--dry-run", action="store_true")
  args = p.parse_args(args=args)

  units.setup(args.units)

  kwargs = {"root": args.root1, "root2": args.root2}

  if args.sampleregex is not None:
    kwargs["filter"] = lambda sample: args.sampleregex.match(sample.SlideID)

  cohort = ZoomCohort(**kwargs)
  if args.dry_run:
    print("would align the following samples:")
    for samp in cohort: print(samp)
  else:
    cohort.run()

if __name__ == "__main__":
  main()
