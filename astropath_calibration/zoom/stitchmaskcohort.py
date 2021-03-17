from ..baseclasses.cohort import DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort
from .stitchmask import StitchInformMask

class StitchMaskCohortBase(DbloadCohort, MaskCohort, SelectRectanglesCohort, WorkflowCohort):
  def __init__(self, *args, filetype=None, **kwargs):
    self.__filetype = filetype
    super().__init__(*args, **kwargs)

  def runsample(self, sample):
    return sample.writemask(filetype=self.__filetype)

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--filetype", choices=(".npz",), default=".npz", help="Format to save the mask.")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "filetype": parsed_args_dict.pop("filetype"),
    }
    return kwargs

class StitchInformMaskCohort(StitchMaskCohortBase):
  __doc__ = StitchInformMask.__doc__

  sampleclass = StitchInformMask

  @property
  def logmodule(self): return "stitchinformmask"

def main(args=None):
  StitchInformMaskCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
