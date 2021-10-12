import abc, methodtools
from .argumentparser import RunFromArgumentParserBase
from .samplemetadata import SampleDef

class MultiCohortBase(RunFromArgumentParserBase):
  @property
  @abc.abstractmethod
  def singlecohortclass(self): pass
  @property
  def globallogentermessage(self): pass

  def __init__(self, roots, **kwargs):
    self.__roots = [pathlib.Path(root) for root in roots]
    self.__kwargs = kwargs

  @methodtools.lru_cache()
  @property
  def cohorts(self):
    return [self.singlecohortclass(root=root, moremainlogroots=self.__roots, **kwargs) for root in self.roots]

  def globallogger(self):
    return MultiLogger(*(c.globallogger for c in self.cohorts), entermessage=self.globallogentermessage)

  @classmethod
  def runfromparsedargs(cls, parsed_args):
    
