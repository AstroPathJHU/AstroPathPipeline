from ...shared.cohort import DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort, XMLPolygonReaderCohort, ZoomFolderCohort
from .annowarpsample import AnnoWarpArgumentParserBase, AnnoWarpArgumentParserTissueMask, AnnoWarpSampleAstroPathTissueMask, AnnoWarpSampleInformTissueMask

class AnnoWarpCohortBase(DbloadCohort, SelectRectanglesCohort, WorkflowCohort, XMLPolygonReaderCohort, ZoomFolderCohort, AnnoWarpArgumentParserBase):
  """
  Cohort for running annowarp over a whole folder of samples.
  """
  def __init__(self, *args, tilepixels=None, **kwargs):
    self.__initiatesamplekwargs = {
      "tilepixels": tilepixels,
    }
    for k, v in list(self.__initiatesamplekwargs.items()):
      if v is None:
        del self.__initiatesamplekwargs[k]
    super().__init__(*args, **kwargs)

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      **self.__initiatesamplekwargs,
    }

  @property
  def workflowkwargs(self):
    return {"layers": [1], **super().workflowkwargs}

class AnnoWarpCohortMask(AnnoWarpCohortBase, MaskCohort, SelectRectanglesCohort, WorkflowCohort, AnnoWarpArgumentParserTissueMask):
  def __init__(self, *args, mintissuefraction=None, **kwargs):
    self.__initiatesamplekwargs = {
      "mintissuefraction": mintissuefraction,
    }
    for k, v in list(self.__initiatesamplekwargs.items()):
      if v is None:
        del self.__initiatesamplekwargs[k]
    super().__init__(*args, **kwargs)

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      **self.__initiatesamplekwargs,
    }


class AnnoWarpCohortInformTissueMask(AnnoWarpCohortMask):
  sampleclass = AnnoWarpSampleInformTissueMask

class AnnoWarpCohortAstroPathTissueMask(AnnoWarpCohortMask):
  sampleclass = AnnoWarpSampleAstroPathTissueMask

def main(args=None):
  AnnoWarpCohortAstroPathTissueMask.runfromargumentparser(args)

if __name__ == "__main__":
  main()
