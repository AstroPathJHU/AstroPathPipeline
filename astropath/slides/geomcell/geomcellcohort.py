from ...baseclasses.cohort import DbloadCohort, GeomFolderCohort, SelectRectanglesCohort, WorkflowCohort
from .geomcellsample import GeomCellSample

class GeomCellCohort(DbloadCohort, GeomFolderCohort, SelectRectanglesCohort, WorkflowCohort):
  sampleclass = GeomCellSample
  __doc__ = sampleclass.__doc__

def main(args=None):
  GeomCellCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
