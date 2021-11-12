from ...shared.argumentparser import CleanupArgumentParser
from ...shared.cohort import DbloadCohort, DeepZoomCohort, SelectLayersCohort, WorkflowCohort, ZoomFolderCohort
from .deepzoomsample import DeepZoomSample

class DeepZoomCohort(DbloadCohort, DeepZoomCohort, SelectLayersCohort, WorkflowCohort, ZoomFolderCohort, CleanupArgumentParser):
  sampleclass = DeepZoomSample
  __doc__ = sampleclass.__doc__

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @property
  def workflowkwargs(self):
    return {"layers": self.layers, "tifflayers": None, **super().workflowkwargs}

def main(args=None):
  DeepZoomCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
