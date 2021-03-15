from ..baseclasses.cohort import DbloadCohort, SelectRectanglesCohort, WorkflowCohort
from .geomsample import GeomSample

class GeomCohort(DbloadCohort, SelectRectanglesCohort, WorkflowCohort):
  sampleclass = GeomSample
  __doc__ = sampleclass.__doc__

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def runsample(self, sample):
    return sample.writeboundaries()

  @property
  def logmodule(self): return "geom"

def main(args=None):
  GeomCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
