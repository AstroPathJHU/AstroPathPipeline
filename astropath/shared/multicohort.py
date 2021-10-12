import abc, methodtools, pathlib
from ..utilities import units
from .argumentparser import InitAndRunFromArgumentParserBase
from .logging import MultiLogger

class MultiCohortBase(InitAndRunFromArgumentParserBase):
  @classmethod
  @abc.abstractmethod
  def singlecohortclass(cls): pass

  @property
  def globallogentermessage(self): pass

  def __init__(self, roots, **initkwargs):
    self.__roots = [pathlib.Path(root) for root in roots]
    self.__initkwargs = initkwargs

  def run(self, **kwargs):
    return [c.run(**kwargs) for c in self.cohorts]
  def dryrun(self, **kwargs):
    for c in self.cohorts:
      c.dryrun(**kwargs)

  @methodtools.lru_cache()
  @property
  def cohorts(self):
    return [self.singlecohortclass(root=root, moremainlogroots=self.__roots, **self.__initkwargs) for root in self.roots]

  def globallogger(self):
    return MultiLogger(*(c.globallogger for c in self.cohorts), entermessage=self.globallogentermessage)

  @classmethod
  def makeargumentparser(cls):
    p = cls.singlecohortclass.makeargumentparser()
    for a in p._actions[:]:
      if a.dest == "root":
        a.nargs = "+"
      elif a.dest.endswith("root"):
        #logroot, dbloadroot, etc. - too complicated to support these at the moment
        p._actions.remove(a)
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    dct = cls.singlecohortclass().initkwargsfromargumentparser(parsed_args_dict)
    assert "roots" not in dct
    dct["roots"] = dct.pop("root")
    return dct
  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    dct = cls.singlecohortclass().runkwargsfromargumentparser(parsed_args_dict)
    return dct
  @classmethod
  def misckwargsfromargumentparser(cls, parsed_args_dict):
    dct = cls.singlecohortclass().misckwargsfromargumentparser(parsed_args_dict)
    return dct

  @classmethod
  def runfromargsdicts(cls, *, initkwargs, runkwargs, misckwargs):
    """
    Run the multicohort from command line arguments.
    """
    with units.setup_context(misckwargs.pop("units")):
      dryrun = misckwargs.pop("dry_run")
      if misckwargs:
        raise TypeError(f"Some miscellaneous kwargs were not processed:\n{misckwargs}")
      if dryrun and "uselogfiles" in initkwargs:
        initkwargs["uselogfiles"] = False
      multicohort = cls(**initkwargs)
      if dryrun:
        multicohort.dryrun(**runkwargs)
      else:
        multicohort.run(**runkwargs)
      return multicohort
