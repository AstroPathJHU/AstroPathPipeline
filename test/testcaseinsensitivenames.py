import git, pathlib, unittest

from .testmarkdownlinks import repo

def lsfilesandfolders(treeorblob):
  yield treeorblob.path
  if isinstance(treeorblob, git.Blob):
    return
  elif isinstance(treeorblob, git.Submodule):
    subrepo = git.Repo(pathlib.PurePosixPath(treeorblob.repo.working_dir)/treeorblob.path)
    for _ in lsfilesandfolders(subrepo.tree()):
      yield pathlib.PurePosixPath(subrepo.working_dir)/_
  elif isinstance(treeorblob, git.Tree):
    for subtreeorblob in treeorblob:
      yield from lsfilesandfolders(subtreeorblob)
  else:
    assert False, type(treeorblob)

class TestCaseInsensitiveNames(unittest.TestCase):
  def testNames(self):
    allpaths = list(lsfilesandfolders(repo().tree()))
    lowercase = [str(_).lower() for _ in allpaths]
    caseinsensitive = {
      lower: [path for path in allpaths if str(path).lower() == lower]
      for lower in lowercase
    }
    bad = []
    total = 0
    for paths in caseinsensitive.values():
      if len(paths) > 1:
        bad.append(", ".join(paths))
      self.assertGreater(len(paths), 0)
      total += len(paths)
    self.assertEqual(len(allpaths), total)
    if bad:
      raise AssertionError("Multiple paths with the same case-insensitive name:\n"+"\n".join(bad))
