from ...shared.argumentparser import CleanupArgumentParser
from ...shared.cohort import DbloadCohort, DeepZoomCohort, SelectLayersCohort, WorkflowCohort, ZoomFolderCohort
from .deepzoomsample import DeepZoomSample, DeepZoomSampleBase, DeepZoomSampleIHC

class DeepZoomCohortBase(DbloadCohort, DeepZoomCohort, WorkflowCohort, ZoomFolderCohort, CleanupArgumentParser):
  sampleclass = DeepZoomSampleBase
  __doc__ = sampleclass.__doc__

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @property
  def workflowkwargs(self):
    return {"tifflayers": None, **super().workflowkwargs}

class DeepZoomCohort(DeepZoomCohortBase, SelectLayersCohort):
  sampleclass = DeepZoomSample
  __doc__ = sampleclass.__doc__

  @property
  def workflowkwargs(self):
    return {"layers": self.layers, **super().workflowkwargs}

class DeepZoomCohortIHC(DeepZoomCohortBase):
  sampleclass = DeepZoomSampleIHC
  __doc__ = sampleclass.__doc__

def main(args=None):
  DeepZoomCohort.runfromargumentparser(args)
def ihc(args=None):
  DeepZoomCohortIHC.runfromargumentparser(args)

if __name__ == "__main__":
  main()
