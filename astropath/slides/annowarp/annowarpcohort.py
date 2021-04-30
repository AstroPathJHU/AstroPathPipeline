from ...baseclasses.cohort import DbloadCohort, MaskCohort, WorkflowCohort, ZoomFolderCohort
from .annowarpsample import AnnoWarpSampleBase, AnnoWarpSampleInformTissueMask

class AnnoWarpCohortBase(DbloadCohort, ZoomFolderCohort, MaskCohort, WorkflowCohort):
  """
  Cohort for running annowarp over a whole folder of samples.

  tilepixels: size of the tiles for alignment (default: 100)
  mintissuefraction: minimum amount of tissue in the tiles to
                     be used for alignment (default: 0.2)
  """
  def __init__(self, *args, tilepixels=None, mintissuefraction=None, **kwargs):
    self.__initiatesamplekwargs = {
      "tilepixels": tilepixels,
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

  sampleclass = AnnoWarpSampleInformTissueMask

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--tilepixels", type=int, default=AnnoWarpSampleInformTissueMask.defaulttilepixels, help=f"size of the tiles to use for alignment (default: {AnnoWarpSampleInformTissueMask.defaulttilepixels})")
    p.add_argument("--min-tissue-fraction", type=float, default=AnnoWarpSampleInformTissueMask.defaultmintissuefraction, help=f"minimum fraction of pixels in the tile that are considered tissue if it's to be used for alignment (default: {AnnoWarpSampleInformTissueMask.defaultmintissuefraction})")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "tilepixels": parsed_args_dict.pop("tilepixels"),
      "mintissuefraction": parsed_args_dict.pop("min_tissue_fraction"),
    }
    return kwargs

class AnnoWarpCohort(AnnoWarpCohortBase):
  def __init__(self, *args, readalignments=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.readalignments = readalignments

  @classmethod
  def argumentparserhelpmessage(cls):
    return AnnoWarpSampleBase.__doc__

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--dont-align", action="store_true", help="read the alignments from existing csv files and just stitch")
    return p

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "readalignments": parsed_args_dict.pop("dont_align"),
    }
    return kwargs

  @property
  def workflowkwargs(self):
    return {"layers": [1], **super().workflowkwargs}

def main(args=None):
  AnnoWarpCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
