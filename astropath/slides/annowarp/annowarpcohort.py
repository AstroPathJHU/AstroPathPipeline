from ...baseclasses.cohort import DbloadCohort, MaskCohort, WorkflowCohort, ZoomFolderCohort
from .annowarpsample import AnnoWarpArgumentParserBase, AnnoWarpSampleBase, AnnoWarpSampleAstroPathTissueMask, AnnoWarpSampleInformTissueMask

class AnnoWarpCohortBase(DbloadCohort, ZoomFolderCohort, MaskCohort, WorkflowCohort, AnnoWarpArgumentParserBase):
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

class AnnoWarpCohortInformTissueMask(AnnoWarpCohortBase):
  sampleclass = AnnoWarpSampleInformTissueMask
  @classmethod
  def maskselectionargumentgroup(cls, argumentparser):
    g = super().maskselectionargumentgroup(argumentparser)
    g.add_argument("--inform-mask", action="store_true", help="use the inform mask found in the component tiff to identify tissue")
    return g
  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    assert parsed_args_dict.pop("inform_mask")
    return super().runkwargsfromargumentparser(parsed_args_dict)

class AnnoWarpCohortAstroPathTissueMask(AnnoWarpCohortBase):
  sampleclass = AnnoWarpSampleAstroPathTissueMask
  @classmethod
  def maskselectionargumentgroup(cls, argumentparser):
    g = super().maskselectionargumentgroup(argumentparser)
    g.add_argument("--astropath-mask", action="store_true", help="use the AstroPath mask to identify tissue")
    return g
  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    assert parsed_args_dict.pop("astropath_mask")
    return super().runkwargsfromargumentparser(parsed_args_dict)

class AnnoWarpCohort(AnnoWarpCohortInformTissueMask, AnnoWarpCohortAstroPathTissueMask):
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
  AnnoWarpCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
