from ...shared.cohort import DbloadCohort, Im3Cohort, SelectRectanglesCohort, WorkflowCohort
from .alignsample import AlignSample, AlignSampleTMA

class AlignCohort(DbloadCohort, Im3Cohort, SelectRectanglesCohort, WorkflowCohort):
  __doc__ = AlignSample.__doc__

  def __init__(self, *args, doalignment=True, dostitching=True, **kwargs):
    super().__init__(*args, **kwargs)
    self.__doalignment = doalignment
    self.__dostitching = dostitching
    if not doalignment and not dostitching:
      raise ValueError("If you do neither alignment nor stitching, there's nothing to do")

  sampleclass = AlignSample
  TMAsampleclass = AlignSampleTMA

  @classmethod
  def makeargumentparser(cls, *, _forworkflow=False, **kwargs):
    p = super().makeargumentparser(_forworkflow=_forworkflow, **kwargs)
    if not _forworkflow:
      g = p.add_mutually_exclusive_group()
      g.add_argument("--dont-align", action="store_true", help="read the alignments from the preexisting align.csv and just do stitching")
      g.add_argument("--dont-stitch", action="store_true", help="skip the stitching step and just do alignment")
    return p

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "doalignment": not parsed_args_dict.pop("dont_align", False),
      "dostitching": not parsed_args_dict.pop("dont_stitch", False),
    }
    return kwargs

  @property
  def workflowkwargs(self) :
    return {
      **super().workflowkwargs,
      "skipannotations": True,  #don't need prepdb annotations output
    }

  @classmethod
  def defaultim3filetype(cls): return "flatWarp"

def main(args=None):
  AlignCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
