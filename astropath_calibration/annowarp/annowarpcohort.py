from ..baseclasses.cohort import DbloadCohort, ZoomCohort
from .annowarpsample import AnnoWarpSample

class AnnoWarpCohort(DbloadCohort, ZoomCohort):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  sampleclass = AnnoWarpSample

  def runsample(self, sample):
    return sample.runannowarp()

  @property
  def logmodule(self): return "annowarp"

def main(args=None):
  AnnoWarpCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
