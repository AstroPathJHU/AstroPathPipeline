from ..baseclasses.cohort import DbloadCohort, FlatwCohort
from .alignmentset import AlignmentSet

class AlignmentCohort(DbloadCohort, FlatwCohort):
  def __init__(self, *args, doalignment=True, dostitching=True, **kwargs):
    super().__init__(*args, **kwargs)
    self.__doalignment = doalignment
    self.__dostitching = dostitching
    if not doalignment and not dostitching:
      raise ValueError("If you do neither alignment nor stitching, there's nothing to do")

  sampleclass = AlignmentSet

  def runsample(self, sample):
    if self.__doalignment:
      sample.getDAPI()
      sample.align()
    else:
      sample.readalignments()

    if self.__dostitching:
      sample.stitch()

  @property
  def logmodule(self): return "align"

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dont-align", action="store_true")
    g.add_argument("--dont-stitch", action="store_true")
    return p

  @classmethod
  def makesampleselectionargumentgroup(cls, parser):
    g = super().makesampleselectionargumentgroup(parser)
    g.add_argument("--skip-aligned", action="store_true")
    return g

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    root = parsed_args_dict["root"]
    dontstitch = parsed_args_dict["dostitching"]
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
    }
    skipaligned = parsed_args_dict.pop("skip_aligned")
    if skipaligned:
      kwargs["filter"] = lambda sample: not (root/sample.SlideID/"dbload"/f"{sample.SlideID}_{'fields' if not dontstitch else 'align'}.csv").exists()
    return kwargs

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "doalignment": not parsed_args_dict.pop("dont_align"),
      "dostitching": not parsed_args_dict.pop("dont_stitch"),
    }
    return kwargs

def main(args=None):
  AlignmentCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
