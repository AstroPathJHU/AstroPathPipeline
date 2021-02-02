from ..baseclasses.cohort import DbloadCohort, ZoomCohort
from .annowarpsample import AnnoWarpSample

class AnnoWarpCohortBase(DbloadCohort, ZoomCohort):
  def __init__(self, *args, tilesize=100, **kwargs):
    self.tilesize = tilesize
    super().__init__(*args, **kwargs)

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, tilesize=self.tilesize}

  sampleclass = AnnoWarpSample

  @property
  def logmodule(self): return "annowarp"

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--tilesize", type=int, default=100)
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "tilesize": parsed_args_dict.pop("tilesize"),
    }
    return kwargs

class AnnoWarpCohort(DbloadCohort, ZoomCohort):
  def runsample(self, sample):
    return sample.runannowarp()

def main(args=None):
  AnnoWarpCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
