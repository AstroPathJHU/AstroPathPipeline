from ...shared.cohort import DbloadCohort, SelectLayersCohort, SelectRectanglesCohort, TempDirCohort, WorkflowCohort, ZoomFolderCohort
from .zoomsample import ZoomSample

class ZoomCohort(DbloadCohort, SelectLayersCohort, SelectRectanglesCohort, TempDirCohort, WorkflowCohort, ZoomFolderCohort):
  __doc__ = ZoomSample.__doc__

  sampleclass = ZoomSample

  def __init__(self, *args, tifflayers, **kwargs):
    self.__tifflayers = tifflayers
    super().__init__(*args, **kwargs)

  @property
  def tifflayers(self): return self.__tifflayers

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "tifflayers": self.tifflayers
    }

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--mode", choices=("vips", "fast", "memmap"), default="vips", help="mode to run zoom: fast is fastest, vips uses the least memory.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tiff-color", action="store_const", default="color", const="color", dest="tiff_layers", help="save false color wsi.tiff (this is the default)")
    g.add_argument("--tiff-layers", type=int, nargs="*", help="layers that go into the output wsi.tiff file")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "tifflayers": parsed_args_dict.pop("tiff_layers"),
    }
    return kwargs

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "mode": parsed_args_dict.pop("mode"),
    }
    return kwargs

  @property
  def workflowkwargs(self):
    return {"layers": self.layers, "tifflayers": self.tifflayers, **super().workflowkwargs}

def main(args=None):
  ZoomCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
