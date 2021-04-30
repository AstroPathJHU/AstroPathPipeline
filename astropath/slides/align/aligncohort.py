from ...baseclasses.cohort import DbloadCohort, Im3Cohort, SelectRectanglesCohort, WorkflowCohort
from .alignsample import AlignSample

class AlignCohort(DbloadCohort, Im3Cohort, SelectRectanglesCohort, WorkflowCohort):
  __doc__ = AlignSample.__doc__

  def __init__(self, *args, doalignment=True, dostitching=True, **kwargs):
    super().__init__(*args, **kwargs)
    self.__doalignment = doalignment
    self.__dostitching = dostitching
    if not doalignment and not dostitching:
      raise ValueError("If you do neither alignment nor stitching, there's nothing to do")

  sampleclass = AlignSample

  def runsample(self, sample):
    if self.__doalignment:
      sample.getDAPI()
      sample.align()
    else:
      sample.readalignments()

    if self.__dostitching:
      sample.stitch()

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dont-align", action="store_true", help="read the alignments from the preexisting align.csv and just do stitching")
    g.add_argument("--dont-stitch", action="store_true", help="skip the stitching step and just do alignment")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "doalignment": not parsed_args_dict.pop("dont_align"),
      "dostitching": not parsed_args_dict.pop("dont_stitch"),
    }
    return kwargs

def main(args=None):
  AlignCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
