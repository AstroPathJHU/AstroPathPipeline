from ...baseclasses.cohort import DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort
from .stitchmask import StitchInformMask

class StitchMaskCohortBase(DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort):
  def runsample(self, sample):
    return sample.writemask()

class StitchInformMaskCohort(StitchMaskCohortBase):
  __doc__ = StitchInformMask.__doc__
  sampleclass = StitchInformMask

def main(args=None):
  StitchInformMaskCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
