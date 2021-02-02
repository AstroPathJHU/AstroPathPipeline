from ..baseclasses.cohort import DbloadCohort, ZoomCohort
from .annowarpsample import AnnoWarpSample

class AnnoWarpCohortBase(DbloadCohort, ZoomCohort):
  def __init__(self, *args, tilesize=100, **kwargs):
    self.tilesize = tilesize
    super().__init__(*args, **kwargs)

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "tilesize": self.tilesize}

  sampleclass = AnnoWarpSample

  @property
  def logmodule(self): return "annowarp"

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--tilesize", type=int, default=100)
    p.add_argument("--skip-stitched", action="store_true")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "tilesize": parsed_args_dict.pop("tilesize"),
    }
    if parsed_args_dict.pop("skip_stitched"):
      dbloadroot = kwargs["dbloadroot"]
      def isnotstitched(sample):
        if not (dbloadroot/sample.SlideID/"{sample}_warp-{kwargs['tilesize']}.csv").exists(): return True
        if not (dbloadroot/sample.SlideID/"{sample}_warp-{kwargs['tilesize']}-stitch.csv").exists(): return True
        return False
      kwargs["filters"].append(isnotstitched)
    return kwargs

class AnnoWarpCohort(DbloadCohort, ZoomCohort):
  def runsample(self, sample):
    return sample.runannowarp()

def main(args=None):
  AnnoWarpCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
