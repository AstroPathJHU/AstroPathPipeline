import doctest, os, pathlib, pkgutil, unittest
import astropath

thisfolder = pathlib.Path(__file__).parent
mainfolder = thisfolder.parent

"""
class TestDocstrings(unittest.TestCase):
  def testdocstrings(self):
    for pythonfile in mainfolder.rglob("*.py"):
      with self.subTest(pythonfile):
        doctest.testfile(os.fspath(pythonfile))
"""
def load_tests(loader, tests, ignore):
  for importer, name, ispkg in pkgutil.walk_packages(astropath.__path__, astropath.__name__ + '.'):
    tests.addTests(doctest.DocTestSuite(name))
  return tests
