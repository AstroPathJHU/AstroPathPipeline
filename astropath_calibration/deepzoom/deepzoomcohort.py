from ..baseclasses.cohort import DbloadCohort, DeepZoomCohort, SelectLayersCohort, SelectRectanglesCohort, ZoomCohort
from .deepzoom import DeepZoomSample

class DeepZoomCohort(DbloadCohort, ZoomCohort, DeepZoomCohort, SelectRectanglesCohort, SelectLayersCohort):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  sampleclass = DeepZoomSample

  def runsample(self, sample):
    return sample.deepzoom()

  @property
  def logmodule(self): return "deepzoom"

def main(args=None):
  DeepZoomCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
