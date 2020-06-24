import argparse, pathlib, re

from ..baseclasses.cohort import FlatwCohort
from ..utilities import units
from .alignmentset import AlignmentSet

class AlignmentCohort(FlatwCohort):
  def initiatesample(self, samp):
    return AlignmentSet(self.root1, self.root2, samp, uselogfiles=True)

  def runsample(self, sample):
    sample.getDAPI()
    sample.align()
    sample.stitch()

  @property
  def logmodule(self): return "align"

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("root1", type=pathlib.Path)
  p.add_argument("root2", type=pathlib.Path)
  p.add_argument("--shred", action="store_true")
  p.add_argument("--debug", action="store_true")
  p.add_argument("--sampleregex", type=re.compile)
  p.add_argument("--units", choices=("safe", "fast"), default="fast")
  args = p.parse_args()

  units.setup(args.units)

  kwargs = {"root1": args.root1, "root2": args.root2, "debug": args.debug}

  if args.sampleregex is not None:
    kwargs["filter"] = lambda sample: args.sampleregex.match(sample.SlideID)

  cohort = AlignmentCohort(**kwargs)
  cohort.run()
