from ...shared.argumentparser import CleanupArgumentParser
from ...shared.cohort import DbloadCohort, GeomFolderCohort, ParallelCohort, SelectRectanglesCohort, WorkflowCohort
from .geomcellsample import GeomCellSampleInform

class GeomCellCohortBase(DbloadCohort, GeomFolderCohort, ParallelCohort, SelectRectanglesCohort, WorkflowCohort, CleanupArgumentParser):
  pass

class GeomCellCohortInform(GeomCellCohortBase):
  sampleclass = GeomCellSampleInform
  __doc__ = sampleclass.__doc__

def inform(args=None):
  GeomCellCohortInform.runfromargumentparser(args)

if __name__ == "__main__":
  inform()
