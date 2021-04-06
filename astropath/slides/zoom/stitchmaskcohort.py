from ...baseclasses.cohort import DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort
from .stitchmask import StitchInformMask

class StitchMaskCohortBase(DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort):
  pass

class StitchInformMaskCohort(StitchMaskCohortBase):
  sampleclass = StitchInformMask
  __doc__ = sampleclass.__doc__

def main(args=None):
  StitchInformMaskCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
