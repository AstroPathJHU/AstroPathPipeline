from ...baseclasses.cohort import DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort
from .stitchmasksample import StitchAstroPathTissueMaskSample, StitchInformMaskSample

class StitchMaskCohortBase(DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort):
  pass

class StitchInformMaskCohort(StitchMaskCohortBase):
  sampleclass = StitchInformMaskSample
  __doc__ = sampleclass.__doc__

class StitchAstroPathTissueMaskCohort(StitchMaskCohortBase):
  sampleclass = StitchAstroPathTissueMaskSample
  __doc__ = sampleclass.__doc__

def informmain(args=None):
  StitchInformMaskCohort.runfromargumentparser(args)

def astropathtissuemain(args=None):
  StitchAstroPathTissueMaskCohort.runfromargumentparser(args)
