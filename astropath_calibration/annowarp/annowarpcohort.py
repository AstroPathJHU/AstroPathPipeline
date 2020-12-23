import argparse, pathlib, re

from ..baseclasses.cohort import DbloadCohort, ZoomCohort
from ..utilities import units
from .annowarpsample import AnnoWarpSample

class AnnoWarpCohort(DbloadCohort, ZoomCohort):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  sampleclass = AnnoWarpSample

  def runsample(self, sample):
    #sample.logger.info(f"{sample.ntiles} {len(sample.rectangles)}")
    return sample.runannowarp()

  @property
  def logmodule(self): return "annowarp"

def main(args=None):
  p = argparse.ArgumentParser()
  p.add_argument("root1", type=pathlib.Path)
  p.add_argument("--debug", action="store_true")
  g = p.add_mutually_exclusive_group()
  g.add_argument("--sampleregex", type=re.compile)
  g.add_argument("--skip-if-wsi-exists", action="store_true")
  p.add_argument("--units", choices=("safe", "fast"), default="fast")
  p.add_argument("--dry-run", action="store_true")
  p.add_argument("--zoom-root", type=pathlib.Path, required=True)
  args = p.parse_args(args=args)

  units.setup(args.units)

  kwargs = {"root": args.root1, "zoomroot": args.zoom_root}

  if args.sampleregex is not None:
    kwargs["filter"] = lambda sample: args.sampleregex.match(sample.SlideID)
  elif args.skip_if_wsi_exists:
    kwargs["filter"] = lambda sample: not all((args.zoom_root/sample.SlideID/"wsi"/(sample.SlideID+f"-Z9-L{layer}-wsi.png")).exists() for layer in range(1, 9))

  cohort = AnnoWarpCohort(**kwargs)
  if args.dry_run:
    print("would annowarp the following samples:")
    for samp in cohort: print(samp)
  else:
    cohort.run()

if __name__ == "__main__":
  main()
