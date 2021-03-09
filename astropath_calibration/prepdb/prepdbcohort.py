from ..baseclasses.cohort import DbloadCohort
from .prepdbsample import PrepdbSample

class PrepdbCohort(DbloadCohort):
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
