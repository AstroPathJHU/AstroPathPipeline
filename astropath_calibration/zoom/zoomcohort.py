from ..baseclasses.cohort import DbloadCohort, SelectRectanglesCohort, ZoomCohort
from .zoom import Zoom

class ZoomCohort(DbloadCohort, SelectRectanglesCohort, ZoomCohort):
  def __init__(self, *args, mode="vips", **kwargs):
    self.__mode = mode
    super().__init__(*args, **kwargs)

  sampleclass = Zoom

  def runsample(self, sample):
    #sample.logger.info(f"{sample.ntiles} {len(sample.rectangles)}")
    return sample.zoom_wsi(mode=self.__mode)

  @property
  def logmodule(self): return "zoom"

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--mode", choices=("vips", "fast", "memmap"), default="vips")
    return p

  @classmethod
  def makesampleselectionargumentgroup(cls, parser):
    g = super().makesampleselectionargumentgroup(parser)
    g.add_argument("--skip-if-wsi-exists", action="store_true")
    return g

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    zoomroot = parsed_args_dict["zoomroot"]
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "mode": parsed_args_dict.pop("mode"),
    }
    skip_if_wsi_exists = parsed_args_dict.pop("skip_if_wsi_exists")
    if skip_if_wsi_exists:
      kwargs["filter"] = lambda sample: not all((zoomroot/sample.SlideID/"wsi"/(sample.SlideID+f"-Z9-L{layer}-wsi.png")).exists() for layer in range(1, 9))
    return kwargs

def main(args=None):
  ZoomCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
