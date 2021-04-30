from ...baseclasses.cohort import DbloadCohort, GeomFolderCohort, PhenotypeFolderCohort, SelectRectanglesCohort, WorkflowCohort
from .csvscansample import CsvScanSample

class CsvScanCohort(DbloadCohort, GeomFolderCohort, PhenotypeFolderCohort, SelectRectanglesCohort, WorkflowCohort):
  sampleclass = CsvScanSample
  __doc__ = sampleclass.__doc__

  def runsample(self, sample):
    return sample.runcsvscan()

def main(args=None):
  CsvScanCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
