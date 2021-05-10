from ...baseclasses.cohort import DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort
from .stitchmasksample import StitchInformMaskSample

class StitchMaskCohortBase(DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort):
  pass

class StitchInformMaskCohort(StitchMaskCohortBase):
  sampleclass = StitchInformMaskSample
  __doc__ = sampleclass.__doc__

def main(args=None):
  StitchInformMaskCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
