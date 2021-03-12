from ..baseclasses.cohort import DbloadCohort, WorkflowCohort
from .prepdbsample import PrepdbSample

class PrepdbCohort(DbloadCohort, WorkflowCohort):
  sampleclass = PrepdbSample
  __doc__ = sampleclass.__doc__

  def runsample(self, sample):
    return sample.writemetadata()

  @property
  def logmodule(self): return "prepdb"

def main(args=None):
  PrepdbCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
