from ...shared.cohort import DbloadCohort, WorkflowCohort
from .prepdbsample import PrepDbArgumentParser, PrepDbSample

class PrepDbCohort(DbloadCohort, WorkflowCohort, PrepDbArgumentParser):
  sampleclass = PrepDbSample
  __doc__ = sampleclass.__doc__

def main(args=None):
  PrepDbCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
