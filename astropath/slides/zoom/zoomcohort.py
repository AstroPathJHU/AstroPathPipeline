from ...baseclasses.cohort import DbloadCohort, SelectLayersCohort, SelectRectanglesCohort, TempDirCohort, WorkflowCohort, ZoomFolderCohort
from .zoomsample import ZoomSample

class ZoomCohort(DbloadCohort, SelectRectanglesCohort, TempDirCohort, ZoomFolderCohort, SelectLayersCohort, WorkflowCohort):
  __doc__ = ZoomSample.__doc__

  sampleclass = ZoomSample

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--mode", choices=("vips", "fast", "memmap"), default="vips", help="mode to run zoom: fast is fastest, vips uses the least memory.")
    return p

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "mode": parsed_args_dict.pop("mode"),
    }
    return kwargs

  @property
  def workflowkwargs(self):
    return {"layers": self.layers, **super().workflowkwargs}

def main(args=None):
  ZoomCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
