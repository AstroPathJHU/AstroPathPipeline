from ..baseclasses.cohort import DbloadCohort, ZoomCohort
from .annowarpsample import AnnoWarpSample

class AnnoWarpCohortBase(DbloadCohort, ZoomCohort):
  def __init__(self, *args, tilepixels=100, **kwargs):
    self.tilepixels = tilepixels
    super().__init__(*args, **kwargs)

  @property
  def initiatesamplekwargs(self):
    return {**super().initiatesamplekwargs, "tilepixels": self.tilepixels}

  sampleclass = AnnoWarpSample

  @property
  def logmodule(self): return "annowarp"

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--tilepixels", type=int, default=100)
    p.add_argument("--skip-stitched", action="store_true")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "tilepixels": parsed_args_dict.pop("tilepixels"),
    }
    if parsed_args_dict.pop("skip_stitched"):
      dbloadroot = kwargs["dbloadroot"]
      def isnotstitched(sample):
        if not (dbloadroot/sample.SlideID/"{sample}_warp-{kwargs['tilepixels']}.csv").exists(): return True
        if not (dbloadroot/sample.SlideID/"{sample}_warp-{kwargs['tilepixels']}-stitch.csv").exists(): return True
        return False
      kwargs["filters"].append(isnotstitched)
    return kwargs

class AnnoWarpCohort(AnnoWarpCohortBase):
  def runsample(self, sample):
    return sample.runannowarp()

def main(args=None):
  AnnoWarpCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
