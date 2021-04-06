from ...baseclasses.cohort import DbloadCohort, SelectRectanglesCohort, WorkflowCohort
from .geomsample import GeomSample

class GeomCohort(DbloadCohort, SelectRectanglesCohort, WorkflowCohort):
  sampleclass = GeomSample
  __doc__ = sampleclass.__doc__

def main(args=None):
  GeomCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
