import argparse, numpy as np, pathlib, re

from ..baseclasses.cohort import FlatwCohort
from ..utilities import units
from .alignmentset import AlignmentSet

class StatsCohort(FlatwCohort):
  def __init__(self, *args, outfile, **kwargs):
    super().__init__(*args, uselogfiles=False, **kwargs)
    self.__outfile = outfile

  def initiatesample(self, samp):
    return AlignmentSet(self.root1, self.root2, samp, uselogfiles=self.uselogfiles)

  def runsample(self, sample):
    sample.readalignments()
    sample.readstitchresult()
    movements = np.array([units.nominal_values(f.pxvec - sample.T@(f.xvec-sample.position)) for f in sample.fields])
    average = np.mean(movements, axis=0)
    rms = np.std(movements, axis=0)
    min = np.min(movements, axis=0)
    max = np.max(movements, axis=0)
    towrite = "{:10} {:10d} {:10.3g} {:10.3g} {:10.3g} {:10.3g} {:10.3g} {:10.3g} {:10.3g} {:10.3g}\n".format(sample.SlideID, len(sample.fields), *average, *rms, *min, *max)
    self.__f.write(towrite)
    sample.logger.info(towrite)

  def run(self, *args, **kwargs):
    with open(self.__outfile, "w") as self.__f:
      self.__f.write("{:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10}\n".format("SlideID", "nfields", "mean x", "mean y", "RMS x", "RMS y", "min x", "min y", "max x", "max y"))
      super().run(*args, **kwargs)

  @property
  def logmodule(self): return "align"

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("root1", type=pathlib.Path)
  p.add_argument("root2", type=pathlib.Path)
  p.add_argument("--debug", action="store_true")
  g = p.add_mutually_exclusive_group()
  g.add_argument("--sampleregex", type=re.compile)
  p.add_argument("--units", choices=("safe", "fast"), default="fast")
  p.add_argument("--dry-run", action="store_true")
  p.add_argument("--outfile")
  args = p.parse_args()

  units.setup(args.units)

  kwargs = {"root": args.root1, "root2": args.root2, "debug": args.debug, "outfile": args.outfile}

  if args.sampleregex is not None:
    kwargs["filter"] = lambda sample: args.sampleregex.match(sample.SlideID)

  cohort = StatsCohort(**kwargs)
  if args.dry_run:
    print("would align the following samples:")
    for samp in cohort: print(samp)
  else:
    cohort.run()
