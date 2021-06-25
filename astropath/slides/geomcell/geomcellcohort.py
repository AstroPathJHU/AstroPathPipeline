from ...shared.cohort import DbloadCohort, GeomFolderCohort, ParallelCohort, SelectRectanglesCohort, WorkflowCohort
from .geomcellsample import GeomCellSample

class GeomCellCohort(DbloadCohort, GeomFolderCohort, ParallelCohort, SelectRectanglesCohort, WorkflowCohort):
  sampleclass = GeomCellSample
  __doc__ = sampleclass.__doc__

def main(args=None):
  GeomCellCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
