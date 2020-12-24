from ..baseclasses.cohort import DbloadCohort, FlatwCohort, ZoomCohort
from .zoom import Zoom

class ZoomCohort(DbloadCohort, FlatwCohort, ZoomCohort):
  def __init__(self, *args, fast=False, **kwargs):
    self.__fast = fast
    super().__init__(*args, **kwargs)

  sampleclass = Zoom

  def runsample(self, sample):
    #sample.logger.info(f"{sample.ntiles} {len(sample.rectangles)}")
    return sample.zoom_wsi(fast=self.__fast)

  @property
  def logmodule(self): return "zoom"

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--fast", action="store_true")
    return p

  @classmethod
  def makesampleselectionargumentgroup(cls, parser):
    g = super().makesampleselectionargumentgroup(parser)
    g.add_argument("--skip-if-wsi-exists", action="store_true")
    return g

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    zoom_root = parsed_args_dict["zoom_root"]
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
    }
    skip_if_wsi_exists = parsed_args_dict.pop("skip_if_wsi_exists")
    if skip_if_wsi_exists:
      kwargs["filter"] = lambda sample: not all((zoom_root/sample.SlideID/"wsi"/(sample.SlideID+f"-Z9-L{layer}-wsi.png")).exists() for layer in range(1, 9))
    return kwargs

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "fast": parsed_args_dict.pop("fast"),
    }
    return kwargs

def main(args=None):
  ZoomCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
