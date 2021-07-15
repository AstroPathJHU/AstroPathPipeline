import doctest, pathlib, pkgutil
import astropath

thisfolder = pathlib.Path(__file__).parent
mainfolder = thisfolder.parent

def load_tests(loader, tests, ignore):
  for importer, name, ispkg in pkgutil.walk_packages(astropath.__path__, astropath.__name__ + '.'):
    tests.addTests(doctest.DocTestSuite(name))
  return tests
