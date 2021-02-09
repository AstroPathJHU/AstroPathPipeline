from ..baseclasses.cohort import DbloadCohort, ZoomCohort
from .annowarpsample import AnnoWarpSample

class AnnoWarpCohortBase(DbloadCohort, ZoomCohort):
  def __init__(self, *args, tilepixels=None, tilebrightnessthreshold=None, mintilebrightfraction=None, mintilerange=None, **kwargs):
    self.__initiatesamplekwargs = {
      "tilepixels": tilepixels,
      "tilebrightnessthreshold": tilebrightnessthreshold,
      "mintilebrightfraction": mintilebrightfraction,
      "mintilerange": mintilerange,
    }
    for k, v in list(self.__initiatesamplekwargs.items()):
      if v is None:
        del self.__initiatesamplekwargs[k]
    super().__init__(*args, **kwargs)

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      **self.__initiatesamplekwargs,
    }

  sampleclass = AnnoWarpSample

  @property
  def logmodule(self): return "annowarp"

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--skip-stitched", action="store_true")
    p.add_argument("--tilepixels", type=int)
    p.add_argument("--tile-brightness-threshold", type=int)
    p.add_argument("--min-tile-bright-fraction", type=float)
    p.add_argument("--min-tile-range", type=int)
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "tilepixels": parsed_args_dict.pop("tilepixels"),
      "tilebrightnessthreshold": parsed_args_dict.pop("tile_brightness_threshold"),
      "mintilebrightfraction": parsed_args_dict.pop("min_tile_bright_fraction"),
      "mintilerange": parsed_args_dict.pop("min_tile_range"),
    }
    if parsed_args_dict.pop("skip_stitched"):
      dbloadroot = kwargs["dbloadroot"]
      def isnotstitched(sample):
        if not (dbloadroot/sample.SlideID/"dbload"/f"{sample.SlideID}_warp-{kwargs['tilepixels']}.csv").exists(): return True
        if not (dbloadroot/sample.SlideID/"dbload"/f"{sample.SlideID}_warp-{kwargs['tilepixels']}-stitch.csv").exists(): return True
        return False
      kwargs["filters"].append(isnotstitched)
    return kwargs

class AnnoWarpCohort(AnnoWarpCohortBase):
  def __init__(self, *args, readalignments=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.readalignments = readalignments

  def runsample(self, sample):
    return sample.runannowarp(readalignments=self.readalignments)

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--dont-align", action="store_true")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "readalignments": parsed_args_dict.pop("dont_align"),
    }
    return kwargs

def main(args=None):
  AnnoWarpCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
