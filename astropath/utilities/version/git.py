import abc, io, methodtools, pathlib, re, subprocess
from .version import astropathversionmatch, have_git
from ..dataclasses import MetaDataAnnotation, MyDataClass
from ..tableio import readtable, writetable

here = pathlib.Path(__file__).parent

_XCHECK_GIT = False

class GitCommand(abc.ABC):
  def __init__(self, repo):
    self.repo = repo
  def __call__(self, *args, **kwargs):
    nogit = self.run_nogit(*args, **kwargs)
    if _XCHECK_GIT and have_git:
      withgit = self.run_git(*args, **kwargs)
      if nogit != withgit:
        raise ValueError(f"Outputs don't match:\n{nogit}\n{withgit}")
    return nogit
  @abc.abstractmethod
  def run_nogit(self, *args, **kwargs): pass
  @abc.abstractmethod
  def run_git(self, *args, **kwargs): pass

class GitRevParse(GitCommand):
  @methodtools.lru_cache()
  def run_git(self, commit):
    return subprocess.run(["git", "rev-parse", commit], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, cwd=self.repo.cwd, encoding="ascii").stdout.strip()

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
    self.initrepo()

  def initrepo(self):
    committables = [_ for _ in (here/"commits_saved.csv", here/"commits.csv") if _.exists()]
    if have_git:
      committables.append(io.StringIO("hash,parents,tags\n"+subprocess.run(["git", "log", "--all", "--pretty=%H\t%P\t%D", "--no-abbrev-commit"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="ascii", check=True, cwd=self.cwd).stdout.replace(",", "").replace("\t", ",")))

    self.commits = set.union(
      *(
        set(readtable(table, GitCommit, extrakwargs={"repo": self}))
        for table in committables
      )
    )

    byhash = {
      hash: {c for c in self.commits if c.hash == hash}
      for hash in {c.hash for c in self.commits}
    }
    for hash, commits in byhash.items():
      while len(commits) > 1:
        commit1, commit2, *_ = commits
        alltags = set(commit1.tags | commit2.tags)

        #grafted means it's the bottom of a shallow clone
        if "grafted" in commit2.tags and "grafted" not in commit1.tags:
          assert not commit1.parents
          parents = commit2.parents
          alltags.remove("grafted")
        elif "grafted" in commit1.tags and "grafted" not in commit2.tags:
          assert not commit2.parents
          parents = commit1.parents
          alltags.remove("grafted")
        else:
          possibleparents = {commit1.parents, commit2.parents}
          try:
            parents, = possibleparents
          except ValueError:
            raise ValueError(f"Inconsistent parents for commit {hash}: {commit1.parents}, {commit2.parents}")

        newcommit = GitCommit(hash=hash, parents=parents, tags=frozenset(alltags))
        for set_ in commits, self.commits:
          set_.remove(commit1)
          set_.remove(commit2)
          set_.add(newcommit)

    for commit in commits:
      if "grafted" in commit.tags:
        raise ValueError(f"No parent info available for commit {commit} --> try using a deeper clone (it doesn't necessarily have to be a full clone) or updating {committables[0]}")

    bytag = {
      tag: {c for c in self.commits if tag in c.tags}
      for tag in frozenset.union(*(c.tags for c in self.commits))
    }
    for tag, commits in bytag.items():
      if len(commits) > 1:
        raise ValueError(f"Multiple commits {[commit.hash for commit in commits]} with the same tag: {tag}")

  def writecommits(self):
    if not have_git:
      raise RuntimeError("Can only write the commits csv if git is available")
    writetable(here/"commits.csv", self.commits)

  @methodtools.lru_cache()
  @property
  def commitdict(self):
    return {
      key: commit for commit in self for key in (commit, commit.hash, *commit.tags)
    }

  def __iter__(self): return iter(self.commits)
  def __len__(self): return len(self.commits)

  @methodtools.lru_cache()
  @property
  def rev_parse(self): return GitRevParse(self)

  @methodtools.lru_cache()
  def getcommit(self, commit, *, _retry=True):
    try:
      return self.commitdict[commit]
    except KeyError:
      if _retry:
        self.initrepo()
        return self.getcommit(commit, _retry=False)
      results = {_ for _ in self if commit == _}
      try:
        result, = results
      except ValueError:
        if not results:
          message = f"Couldn't find a commit matching {commit}."
          if not have_git:
            message += " Maybe try running python setup.py build_commits_csv"
          raise ValueError(message)
        else:
          raise ValueError(f"Found multiple commits matching {commit}: " + ", ".join(str(_) for _ in results))
      return result

  @property
  def currentcommit(self):
    return self.getcommit(astropathversionmatch.group("commit"))

class GitCommit(MyDataClass):
  hash: str
  @property
  def parents(self):
    return tuple(self.repo.getcommit(p) for p in self.__parents)
  @parents.setter
  def parents(self, parents):
    self.__parents = parents
  parents: tuple = MetaDataAnnotation(
    parents,
    writefunction=lambda x: " ".join(_.hash for _ in x),
    readfunction=lambda x: tuple(x.split()),
    usedefault=False
  )
  tags: frozenset = MetaDataAnnotation(
    writefunction=lambda x: " ".join(sorted(x)),
    readfunction=lambda x: frozenset(
      _ for _ in x.split()
      if re.match("v[0-9]+[.][0-9]+[.][0-9]+", _)
      or _ == "grafted"
    ),
  )
  repo: GitRepo = MetaDataAnnotation(includeintable=False)

  def __copy__(self):
    return GitCommit(hash=self.hash, parents=self.__parents, tags=self.tags, repo=self.repo)
  def __deepcopy__(self, memo):
    return GitCommit(hash=self.hash, parents=self.__parents, tags=self.tags, repo=self.repo)
  def __eq__(self, other):
    if isinstance(other, GitCommit):
      return self.hash == other.hash
    if self.hash == other: return True
    return self.hash == self.repo.rev_parse(other)
  def __lt__(self, other):
    if not isinstance(other, GitCommit): return NotImplemented
    return self.isancestor(other) and self != other
  def __le__(self, other):
    if not isinstance(other, GitCommit): return NotImplemented
    return self.isancestor(other)
  def __gt__(self, other):
    if not isinstance(other, GitCommit): return NotImplemented
    return other < self
  def __ge__(self, other):
    if not isinstance(other, GitCommit): return NotImplemented
    return other <= self
  def __str__(self):
    return self.hash
  @methodtools.lru_cache()
  @property
  def recursiveparents(self):
    result = set()
    tohandle = {self}
    while tohandle:
      result |= tohandle
      tohandle = frozenset.union(*(frozenset(_.parents) for _ in tohandle)) - result
    return frozenset(result)
  def shorthash(self, length=8):
    return self.hash[:length]
  @methodtools.lru_cache()
  def isancestor(self, other):
    other = self.repo.getcommit(other)
    return self in other.recursiveparents
  def __hash__(self):
    return hash(self.hash)
  def __repr__(self):
    return f"{type(self).__name__}(hash={self.hash!r}, parents={self.__parents!r}, tags={self.tags!r}, repo={self.repo!r})"

thisrepo = GitRepo(here.parent.parent.parent)
