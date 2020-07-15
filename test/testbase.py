import abc, contextlib, dataclasses, numbers, numpy as np, pathlib, shutil, tempfile, unittest

from ..utilities import units

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
    cls.__output = None

  @abc.abstractproperty
  def outputfilenames(self): pass

  def saveoutput(self):
    if self.__output is not None: return
    type(self).__output = contextlib.ExitStack()
    self.__output.__enter__()
    for filename in self.outputfilenames:
      self.__output.enter_context(temporarilyremove(filename))

  def setUp(self):
    self.maxDiff = None
    for filename in self.outputfilenames:
      try:
        filename.unlink()
      except FileNotFoundError:
        pass

  def tearDown(self):
    pass

  @classmethod
  def tearDownClass(cls):
    if cls.__output is not None:
      cls.__output.__exit__(None, None, None)
