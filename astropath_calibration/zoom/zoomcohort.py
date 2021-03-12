from ..baseclasses.cohort import DbloadCohort, SelectLayersCohort, SelectRectanglesCohort, TempDirCohort, ZoomCohort
from .zoom import Zoom

class ZoomCohort(DbloadCohort, SelectRectanglesCohort, TempDirCohort, ZoomCohort, SelectLayersCohort):
  __doc__ = Zoom.__doc__

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
    p.add_argument("--mode", choices=("vips", "fast", "memmap"), default="vips", help="mode to run zoom: fast is fastest, vips uses the least memory.")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "mode": parsed_args_dict.pop("mode"),
    }
    return kwargs

def main(args=None):
  ZoomCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
