import numpy as np, pathlib

from ...baseclasses.cohort import DbloadCohort, Im3Cohort
from ...utilities import units
from .alignmentset import AlignmentSet

class StatsCohort(DbloadCohort, Im3Cohort):
  def __init__(self, *args, outfile, **kwargs):
    super().__init__(*args, uselogfiles=False, **kwargs)
    self.__outfile = outfile

  sampleclass = AlignmentSet

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

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--outfile", type=pathlib.Path, required=True)
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "outfile": parsed_args_dict.pop("outfile")
    }

def main(args=None):
  StatsCohort.runfromargumentparser(args)
