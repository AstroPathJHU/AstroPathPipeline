from ...shared.cohort import DbloadCohort, XMLPolygonReaderCohort, WorkflowCohort
from .prepdbsample import PrepDbArgumentParser, PrepDbSample

class PrepDbCohort(DbloadCohort, WorkflowCohort, XMLPolygonReaderCohort, PrepDbArgumentParser):
  sampleclass = PrepDbSample
  __doc__ = sampleclass.__doc__

  def __init__(self, *args, margin, annotationsonwsi, annotationposition, **kwargs):
    super().__init__(*args, **kwargs)
    self.__margin = margin
    self.__annotationsonwsi = annotationsonwsi
    self.__annotationposition = annotationposition

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "margin": self.__margin,
      "annotationsonwsi": self.__annotationsonwsi,
      "annotationposition": self.__annotationposition,
    }

def main(args=None):
  PrepDbCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
