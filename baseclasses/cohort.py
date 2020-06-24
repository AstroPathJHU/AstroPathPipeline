import abc, pathlib, traceback
from ..utilities.tableio import readtable
from .logging import getlogger
from .sample import SampleDef

class Cohort(abc.ABC):
  def __init__(self, root, *, filter=lambda samp: True, debug=False):
    super().__init__()
    self.root = pathlib.Path(root)
    self.filter = filter
    self.debug = debug

  def __iter__(self):
    for samp in readtable(self.root1/"sampledef.csv", SampleDef):
      if not samp: continue
      if not self.filter(samp): continue
      yield samp

  @abc.abstractmethod
  def runsample(self, sample):
    "actually run whatever is supposed to be run on the sample"

  @abc.abstractmethod
  def initiatesample(self, samp):
    "Create a Sample object (subclass of SampleBase) from SampleDef samp to run on"

  @abc.abstractproperty
  def logmodule(self):
    "name of the log files for this class (e.g. align)"

  def run(self):
    for samp in self:
      with getlogger(self.logmodule, self.root, samp, uselogfiles=True) as logger:
        try:
          sample = self.initiatesample(samp)
          if sample.logmodule != self.logmodule:
            raise ValueError(f"Wrong logmodule: {self.logmodule} != {sample.logmodule}")
          with sample:
            self.runsample(sample)
        except Exception as e:
          logger.error(str(e).replace(";", ","))
          logger.info(repr(traceback.format_exc()).replace(";", ""))
          if self.debug: raise

class FlatwCohort(Cohort):
  def __init__(self, root, root2, *args, **kwargs):
    super().__init__(root=root, *args, **kwargs)
    self.root2 = pathlib.Path(root2)

  @property
  def root1(self): return self.root
