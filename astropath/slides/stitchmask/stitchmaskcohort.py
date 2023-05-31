from ...shared.cohort import DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort
from .stitchmasksample import StitchAstroPathTissueMaskSample, StitchAstroPathTissueMaskSampleTMA, StitchIHCTissueMaskSample, StitchInformMaskSample

class StitchMaskCohortBase(DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort):
  pass

class StitchInformMaskCohort(StitchMaskCohortBase):
  sampleclass = StitchInformMaskSample
  __doc__ = sampleclass.__doc__

class StitchAstroPathTissueMaskCohort(StitchMaskCohortBase):
  sampleclass = StitchAstroPathTissueMaskSample
  TMAsampleclass = StitchAstroPathTissueMaskSampleTMA
  __doc__ = sampleclass.__doc__

  @property
  def workflowkwargs(self):
    return {**super().workflowkwargs, "skip_masking": False}

class StitchIHCTissueMaskCohort(StitchMaskCohortBase):
  sampleclass = StitchIHCTissueMaskSample
  __doc__ = sampleclass.__doc__

def informmain(args=None):
  StitchInformMaskCohort.runfromargumentparser(args)

def astropathtissuemain(args=None):
  StitchAstroPathTissueMaskCohort.runfromargumentparser(args)

def ihcmain(args=None):
  StitchIHCTissueMaskCohort.runfromargumentparser(args)
