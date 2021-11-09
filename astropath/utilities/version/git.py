import abc, io, methodtools, pathlib, subprocess
from .version import astropathversionmatch, have_git
from ..dataclasses import MetaDataAnnotation, MyDataClass
from ..tableio import readtable, writetable

here = pathlib.Path(__file__).parent

class GitCommand(abc.ABC):
  def __init__(self, repo):
    self.repo = repo
  def __call__(self, *args, **kwargs):
    nogit = self.run_nogit(*args, **kwargs)
    if not have_git:
      withgit = self.run_git(*args, **kwargs)
      if nogit != withgit:
        raise ValueError(f"Outputs don't match:\n{nogit}\n{withgit}")
  @abc.abstractmethod
  def run_nogit(self, *args, **kwargs): pass
  @abc.abstractmethod
  def run_git(self, *args, **kwargs): pass

class GitRevParse(GitCommand):
  @methodtools.lru_cache()
  def run_git(self, commit):
    return subprocess.run(["git", "rev-parse", commit], capture_output=True, check=True, cwd=self.repo.cwd, encoding="ascii").stdout.strip()

  @methodtools.lru_cache()
  def run_nogit(self, commit):
    result = set()
    for othercommit in self.repo:
      if othercommit.hash.startswith(commit) or commit in othercommit.tags:
        result.add(othercommit.hash)
    try:
      result, = result
    except ValueError:
      if result:
        raise ValueError(f"Multiple commits match {commit}:\n{result}")
      else:
        raise ValueError(f"No commits match {commit}")
    return result

class GitRepo:
  def __init__(self, cwd):
    self.cwd = cwd

    if have_git:
      committable = io.StringIO("hash,parents,tags\n"+subprocess.run(["git", "log", "--all", "--pretty=%H,%P,%D", "--no-abbrev-commit"], capture_output=True, encoding="ascii").stdout)
    else:
      committable = here/"commits.csv"

    self.commits = frozenset(readtable(committable, GitCommit, extrakwargs={"repo": self}))

  def writecommits(self):
    if not have_git:
      raise RuntimeError("Can only write the commits csv if git is available")
    writetable(here/"commits.csv", self.commits)

  def __iter__(self): return iter(self.commits)

  @methodtools.lru_cache()
  @property
  def rev_parse(self): return GitRevParse(self)

  def currentcommit(self):
    return self.rev_parse(astropathversionmatch.group("commit"))

class GitCommit(MyDataClass):
  hash: str
  parents: frozenset = MetaDataAnnotation(writefunction=lambda x: ",".join(sorted(x)), readfunction=lambda x: frozenset(x.split()))
  tags: frozenset = MetaDataAnnotation(writefunction=lambda x: ",".join(sorted(x)), readfunction=lambda x: frozenset(_ for _ in x.split() if _ != "->"))
  repo: GitRepo = MetaDataAnnotation(includeintable=False)

  def __eq__(self, other):
    if isinstance(other, GitCommit):
      return self.hash == other.hash
    if self.hash == other: return True
    return self.hash == self.repo.rev_parse(other)
  @methodtools.lru_cache()
  def isancestor(self, other):
    return other == self or any(self.isancestor(parent) for parent in other.parents)
  def __hash__(self):
    return hash(self.hash)

