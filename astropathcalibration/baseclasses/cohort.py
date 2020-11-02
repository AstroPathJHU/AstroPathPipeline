import abc, pathlib
from ..utilities.tableio import readtable
from .logging import getlogger
from .sample import SampleDef

class Cohort(abc.ABC):
  def __init__(self, root, *, filter=lambda samp: True, debug=False, uselogfiles=True):
    super().__init__()
    self.root = pathlib.Path(root)
    self.filter = filter
    self.debug = debug
    self.uselogfiles = uselogfiles

  def __iter__(self):
    for samp in readtable(self.root/"sampledef.csv", SampleDef):
      if not samp: continue
      if not self.filter(samp): continue
      yield samp

  @abc.abstractmethod
  def runsample(self, sample, **kwargs):
    "actually run whatever is supposed to be run on the sample"

  @abc.abstractmethod
  def initiatesample(self, samp, **kwargs):
    "Create a Sample object (subclass of SampleBase) from SampleDef samp to run on"

  @abc.abstractproperty
  def logmodule(self):
    "name of the log files for this class (e.g. align)"

  def run(self, **kwargs):
    for samp in self:
      with getlogger(module=self.logmodule, root=self.root, samp=samp, uselogfiles=self.uselogfiles, reraiseexceptions=self.debug):  #log exceptions in __init__ of the sample
        sample = self.initiatesample(samp, reraiseexceptions=self.debug)
        if sample.logmodule != self.logmodule:
          raise ValueError(f"Wrong logmodule: {self.logmodule} != {sample.logmodule}")
        with sample:
          self.runsample(sample, **kwargs)

class FlatwCohort(Cohort):
  def __init__(self, root, root2, *args, **kwargs):
    super().__init__(root=root, *args, **kwargs)
    self.root2 = pathlib.Path(root2)

  @property
  def root1(self): return self.root
