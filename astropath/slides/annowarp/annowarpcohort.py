from ...shared.cohort import DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort, XMLPolygonReaderCohort, ZoomFolderCohort
from .annowarpsample import AnnoWarpArgumentParserBase, AnnoWarpArgumentParserAstroPathTissueMask, AnnoWarpArgumentParserInformTissueMask, AnnoWarpArgumentParserTissueMask, AnnoWarpSampleAstroPathTissueMask, AnnoWarpSampleInformTissueMask

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


class AnnoWarpCohortInformTissueMask(AnnoWarpCohortMask, AnnoWarpArgumentParserInformTissueMask):
  sampleclass = AnnoWarpSampleInformTissueMask

class AnnoWarpCohortAstroPathTissueMask(AnnoWarpCohortMask, AnnoWarpArgumentParserAstroPathTissueMask):
  sampleclass = AnnoWarpSampleAstroPathTissueMask

class AnnoWarpCohortSelectMask(AnnoWarpCohortInformTissueMask, AnnoWarpCohortAstroPathTissueMask):
  def __init__(self, *args, **kwargs):
    raise TypeError("This class should not be instantiated")
  sampleclass = None
  @classmethod
  def defaultunits(cls):
    if cls is AnnoWarpCohortSelectMask:
      result, = {_.defaultunits() for _ in cls.__bases__}
      return result
    return super().defaultunits()
  @classmethod
  def runfromargumentparser(cls, args=None):
    p = cls.makeargumentparser()
    parsed_args = p.parse_args(args=args)
    if parsed_args.inform_mask:
      return AnnoWarpCohortInformTissueMask.runfromargumentparser(args=args)
    elif parsed_args.astropath_mask:
      return AnnoWarpCohortAstroPathTissueMask.runfromargumentparser(args=args)
    else:
      assert False

def main(args=None):
  AnnoWarpCohortSelectMask.runfromargumentparser(args)

if __name__ == "__main__":
  main()
