from ..baseclasses.cohort import DbloadCohort, DeepZoomCohort, SelectLayersCohort, WorkflowCohort, ZoomCohort
from .deepzoom import DeepZoomSample

class DeepZoomCohort(DbloadCohort, ZoomCohort, DeepZoomCohort, SelectLayersCohort, WorkflowCohort):
  sampleclass = DeepZoomSample
  __doc__ = sampleclass.__doc__

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def runsample(self, sample):
    return sample.deepzoom()

def main(args=None):
  DeepZoomCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
