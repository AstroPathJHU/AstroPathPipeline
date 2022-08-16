from ...shared.cohort import DbloadCohort, WorkflowCohort
from .prepdbsample import PrepDbArgumentParser, PrepDbSample

class PrepDbCohort(DbloadCohort, WorkflowCohort, PrepDbArgumentParser):
  sampleclass = PrepDbSample
  __doc__ = sampleclass.__doc__

  def __init__(self, *args, margin, **kwargs):
    super().__init__(*args, **kwargs)
    self.__margin = margin

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "margin": self.__margin,
    }

def main(args=None):
  PrepDbCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
