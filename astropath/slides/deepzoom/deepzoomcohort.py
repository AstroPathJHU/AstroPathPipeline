from ...shared.cohort import DbloadCohort, DeepZoomCohort, SelectLayersCohort, WorkflowCohort, ZoomFolderCohort
from .deepzoomsample import DeepZoomSample

class DeepZoomCohort(DbloadCohort, DeepZoomCohort, SelectLayersCohort, WorkflowCohort, ZoomFolderCohort):
  sampleclass = DeepZoomSample
  __doc__ = sampleclass.__doc__

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @property
  def workflowkwargs(self):
    return {"layers": self.layers, **super().workflowkwargs}

def main(args=None):
  DeepZoomCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
