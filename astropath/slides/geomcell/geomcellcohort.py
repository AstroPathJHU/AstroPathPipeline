from ...shared.argumentparser import CleanupArgumentParser
from ...shared.cohort import DbloadCohort, GeomFolderCohort, ParallelCohort, SegmentationFolderCohort, SelectRectanglesCohort, WorkflowCohort
from .geomcellsample import GeomCellSampleDeepCell, GeomCellSampleDeepCellBase, GeomCellSampleInform, GeomCellSampleMesmer

class GeomCellCohortBase(DbloadCohort, GeomFolderCohort, ParallelCohort, SelectRectanglesCohort, WorkflowCohort, CleanupArgumentParser):
  pass

class GeomCellCohortInform(GeomCellCohortBase):
  sampleclass = GeomCellSampleInform
  __doc__ = sampleclass.__doc__

class GeomCellCohortDeepCellBase(GeomCellCohortBase, SegmentationFolderCohort):
  sampleclass = GeomCellSampleDeepCellBase

class GeomCellCohortDeepCell(GeomCellCohortDeepCellBase):
  sampleclass = GeomCellSampleDeepCell
  __doc__ = sampleclass.__doc__

class GeomCellCohortMesmer(GeomCellCohortDeepCellBase):
  sampleclass = GeomCellSampleMesmer
  __doc__ = sampleclass.__doc__

def inform(args=None):
  GeomCellCohortInform.runfromargumentparser(args)
def deepcell(args=None):
  GeomCellCohortDeepCell.runfromargumentparser(args)
def mesmer(args=None):
  GeomCellCohortMesmer.runfromargumentparser(args)

if __name__ == "__main__":
  inform()
