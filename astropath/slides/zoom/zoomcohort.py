from ...shared.cohort import DbloadCohort, MaskCohort, SelectLayersCohort, SelectRectanglesCohort, TempDirCohort, WorkflowCohort, ZoomFolderCohort
from .zoomsample import ZoomArgumentParserBase, ZoomArgumentParserComponentTiff, ZoomSample, ZoomSampleBase, ZoomSampleIHC, ZoomSampleTMA

class ZoomCohortBase(DbloadCohort, MaskCohort, SelectRectanglesCohort, TempDirCohort, WorkflowCohort, ZoomFolderCohort, ZoomArgumentParserBase):
  sampleclass = ZoomSampleBase

  def __init__(self, *args, tifflayers, **kwargs):
    self.__tifflayers = tifflayers
    super().__init__(*args, **kwargs)

  @property
  def tifflayers(self): return self.__tifflayers

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "tifflayers": self.tifflayers
    }

  @property
  def workflowkwargs(self):
    return {"tifflayers": self.tifflayers, **super().workflowkwargs}

class ZoomCohort(ZoomCohortBase, SelectLayersCohort, ZoomArgumentParserComponentTiff):
  __doc__ = ZoomSample.__doc__

  sampleclass = ZoomSample
  sampleclassTMA = ZoomSampleTMA
  defaulttifflayers = "color"

  @property
  def workflowkwargs(self):
    return {"layers": self.layers, **super().workflowkwargs}

class ZoomCohortIHC(ZoomCohortBase):
  __doc__ = ZoomSample.__doc__

  sampleclass = ZoomSampleIHC

def main(args=None):
  ZoomCohort.runfromargumentparser(args)
def ihc(args=None):
  ZoomCohortIHC.runfromargumentparser(args)

if __name__ == "__main__":
  main()
