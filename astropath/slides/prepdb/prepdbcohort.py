from ...baseclasses.cohort import DbloadCohort, WorkflowCohort
from .prepdbsample import PrepDbSample

class PrepDbCohort(DbloadCohort, WorkflowCohort):
  sampleclass = PrepDbSample
  __doc__ = sampleclass.__doc__

  def runsample(self, sample):
    return sample.writemetadata()

def main(args=None):
  PrepDbCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
