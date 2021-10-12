import abc, methodtools, pathlib
from ..utilities import units
from .argumentparser import ArgumentParserWithVersionRequirement
from .logging import MultiLogger

class MultiCohortBase(ArgumentParserWithVersionRequirement):
  @classmethod
  @abc.abstractmethod
  def singlecohortclass(cls): pass

  @property
  def globallogentermessage(self): pass

  def __init__(self, root, logroot=None, moremainlogroots=[], **initkwargs):
    initkwargs["root"] = root
    initkwargs["logroot"] = logroot

    percohortkws = set()

    for kw, kwarg in initkwargs.copy().items():
      if "root" in kw:
        percohortkws.add(kw)
        if kwarg is None:
          initkwargs[kw] = [None for _ in root]
        else:
          if len(kwarg) != len(root):
            raise ValueError(f"Provided {len(root)} roots, but {len(kwarg)} {kw}s")

    logroot = initkwargs["logroot"]
    if frozenset(logroot) == {None}: logroot = initkwargs["root"]
    initkwargs["moremainlogroots"] = frozenset(moremainlogroots) | frozenset(logroot)

    self.__initkwargs = []
    for i, _ in enumerate(root):
      kwargs = initkwargs.copy()
      self.__initkwargs.append(kwargs)
      for kw, kwarg in kwargs.items():
        if kw in percohortkws:
          assert len(kwarg) == len(root)
          kwargs[kw] = kwarg[i]

  def run(self, **kwargs):
    return [c.run(**kwargs) for c in self.cohorts]
  def dryrun(self, **kwargs):
    for c in self.cohorts:
      c.dryrun(**kwargs)

  @methodtools.lru_cache()
  @property
  def cohorts(self):
    return [self.singlecohortclass(**initkwargs) for initkwargs in self.__initkwargs]

  def globallogger(self, **kwargs):
    return MultiLogger(*(c.globallogger(**kwargs) for c in self.cohorts), entermessage=self.globallogentermessage)

  @classmethod
  def makeargumentparser(cls):
    p = cls.singlecohortclass.makeargumentparser()
    for a in p._actions[:]:
      if "root" in a.dest:
        assert a.nargs in {1, None}, a.nargs
        a.nargs = "+"
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    dct = cls.singlecohortclass.initkwargsfromargumentparser(parsed_args_dict)
    return dct
  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    dct = cls.singlecohortclass.runkwargsfromargumentparser(parsed_args_dict)
    return dct
  @classmethod
  def misckwargsfromargumentparser(cls, parsed_args_dict):
    dct = cls.singlecohortclass.misckwargsfromargumentparser(parsed_args_dict)
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
