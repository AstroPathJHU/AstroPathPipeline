from ...shared.cohort import DbloadCohort, MaskCohort, WorkflowCohort, XMLPolygonReaderCohort, ZoomFolderCohort
from .annowarpsample import AnnoWarpArgumentParserBase, AnnoWarpArgumentParserAstroPathTissueMask, AnnoWarpArgumentParserInformTissueMask, AnnoWarpSampleAstroPathTissueMask, AnnoWarpSampleInformTissueMask

class AnnoWarpCohortBase(DbloadCohort, ZoomFolderCohort, MaskCohort, WorkflowCohort, XMLPolygonReaderCohort, AnnoWarpArgumentParserBase):
  """
  Cohort for running annowarp over a whole folder of samples.

  tilepixels: size of the tiles for alignment (default: 100)
  mintissuefraction: minimum amount of tissue in the tiles to
                     be used for alignment (default: 0.2)
  """
  def __init__(self, *args, tilepixels=None, mintissuefraction=None, readalignments=False, **kwargs):
    self.__initiatesamplekwargs = {
      "tilepixels": tilepixels,
      "mintissuefraction": mintissuefraction,
    }
    for k, v in list(self.__initiatesamplekwargs.items()):
      if v is None:
        del self.__initiatesamplekwargs[k]
    self.readalignments = readalignments
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

class AnnoWarpCohortInformTissueMask(AnnoWarpCohortBase, AnnoWarpArgumentParserInformTissueMask):
  sampleclass = AnnoWarpSampleInformTissueMask

class AnnoWarpCohortAstroPathTissueMask(AnnoWarpCohortBase, AnnoWarpArgumentParserAstroPathTissueMask):
  sampleclass = AnnoWarpSampleAstroPathTissueMask

class AnnoWarpCohortSelectMask(AnnoWarpCohortInformTissueMask, AnnoWarpCohortAstroPathTissueMask):
  def __init__(self, *args, **kwargs):
    raise TypeError("This class should not be instantiated")
  sampleclass = None
  @classmethod
  def defaultunits(cls):
    result, = {_.defaultunits() for _ in cls.__bases__}
    return result
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
