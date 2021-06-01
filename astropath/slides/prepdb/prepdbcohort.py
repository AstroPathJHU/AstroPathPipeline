from ...shared.cohort import DbloadCohort, XMLPolygonReaderCohort, WorkflowCohort
from .prepdbsample import PrepDbArgumentParser, PrepDbSample

class PrepDbCohort(DbloadCohort, WorkflowCohort, XMLPolygonReaderCohort, PrepDbArgumentParser):
  sampleclass = PrepDbSample
  __doc__ = sampleclass.__doc__

def main(args=None):
  PrepDbCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
