import argparse, pathlib, re

from ..baseclasses.cohort import FlatwCohort
from ..utilities import units
from .alignmentset import AlignmentSet

class AlignmentCohort(FlatwCohort):
  def __init__(self, *args, doalignment=True, dostitching=True, **kwargs):
    super().__init__(*args, **kwargs)
    self.__doalignment = doalignment
    self.__dostitching = dostitching
    if not doalignment and not dostitching:
      raise ValueError("If you do neither alignment nor stitching, there's nothing to do")

  def initiatesample(self, samp, **kwargs):
    return AlignmentSet(self.root1, self.root2, samp, uselogfiles=self.uselogfiles, **kwargs)

  def runsample(self, sample):
    if self.__doalignment:
      sample.getDAPI()
      sample.align()
    else:
      sample.readalignments()

    if self.__dostitching:
      sample.stitch()

  @property
  def logmodule(self): return "align"

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
  g = p.add_mutually_exclusive_group()
  g.add_argument("--dont-align", action="store_true")
  g.add_argument("--dont-stitch", action="store_true")
  args = p.parse_args(args=args)

  units.setup(args.units)

  kwargs = {"root": args.root1, "root2": args.root2, "debug": args.debug}

  if args.sampleregex is not None:
    kwargs["filter"] = lambda sample: args.sampleregex.match(sample.SlideID)
  elif args.skip_aligned:
    kwargs["filter"] = lambda sample: not (args.root1/sample.SlideID/"dbload"/(sample.SlideID+"_fields.csv")).exists()

  kwargs["doalignment"] = not args.dont_align
  kwargs["dostitching"] = not args.dont_stitch

  cohort = AlignmentCohort(**kwargs)
  if args.dry_run:
    print("would align the following samples:")
    for samp in cohort: print(samp)
  else:
    cohort.run()

if __name__ == "__main__":
  main()
