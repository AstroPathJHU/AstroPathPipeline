import abc, contextlib, dataclasses, numbers, numpy as np, pathlib, shutil, tempfile, unittest

from astropathcalibration.utilities import units

def assertAlmostEqual(a, b, **kwargs):
  if isinstance(a, units.safe.Distance):
    return units.np.testing.assert_allclose(a, b, **kwargs)
  elif isinstance(a, numbers.Number):
    if isinstance(b, units.safe.Distance): b = float(b)
    return np.testing.assert_allclose(a, b, **kwargs)
  elif dataclasses.is_dataclass(type(a)) and type(a) == type(b):
    try:
      for field in dataclasses.fields(type(a)):
        assertAlmostEqual(getattr(a, field.name), getattr(b, field.name), **kwargs)
    except AssertionError:
      np.testing.assert_equal(a, b)
  else:
    return np.testing.assert_equal(a, b)

def expectedFailureIf(condition):
  if condition:
    return unittest.expectedFailure
  else:
    return lambda function: function

@contextlib.contextmanager
def temporarilyremove(filepath):
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    tmppath = d/filepath.name
    shutil.move(filepath, tmppath)
    try:
      yield
    finally:
      shutil.move(tmppath, filepath)

@contextlib.contextmanager
def temporarilyreplace(filepath, temporarycontents):
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    tmppath = d/filepath.name
    shutil.move(filepath, tmppath)
    with open(filepath, "w") as f:
      f.write(temporarycontents)
    try:
      yield
    finally:
      shutil.move(tmppath, filepath)

class TestBaseSaveOutput(abc.ABC, unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.__output = contextlib.ExitStack()
    cls.__output.__enter__()
    cls.__saved = set()

  @abc.abstractproperty
  def outputfilenames(self): pass

  def saveoutput(self):
    for filename in self.outputfilenames:
      if filename not in self.__saved and filename.exists():
        self.__saved.add(filename)
        self.__output.enter_context(temporarilyremove(filename))

  def removeoutput(self):
    for filename in self.outputfilenames:
      try:
        filename.unlink()
      except FileNotFoundError:
        pass

  def setUp(self):
    self.maxDiff = None
    self.removeoutput()

  def tearDown(self):
    pass

  @classmethod
  def tearDownClass(cls):
    cls.__output.__exit__(None, None, None)
