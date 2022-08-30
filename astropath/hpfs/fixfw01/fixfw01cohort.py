from ...shared.cohort import DbloadCohort, Im3Cohort, SelectLayersIm3Cohort, SelectRectanglesCohort, WorkflowCohort
from .fixfw01sample import FixFW01ArgumentParser, FixFW01SampleBase, FixFW01SampleDbload, FixFW01SampleXML

class FixFW01CohortBase(Im3Cohort, SelectLayersIm3Cohort, SelectRectanglesCohort, WorkflowCohort, FixFW01ArgumentParser):
  sampleclass = FixFW01SampleBase
  __doc__ = sampleclass.__doc__

  def __init__(self, *args, layers=None, **kwargs):
    if layers is None: layers = [1]
    super().__init__(*args, layers=layers, **kwargs)

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
    }

class FixFW01CohortXML(FixFW01CohortBase):
  sampleclass = FixFW01SampleXML
  __doc__ = sampleclass.__doc__

class FixFW01CohortDbload(FixFW01CohortBase, DbloadCohort):
  sampleclass = FixFW01SampleDbload
  __doc__ = sampleclass.__doc__

def main_xml(args=None):
  FixFW01CohortXML.runfromargumentparser(args)
def main_dbload(args=None):
  FixFW01CohortDbload.runfromargumentparser(args)

if __name__ == "__main__":
  main_xml()
