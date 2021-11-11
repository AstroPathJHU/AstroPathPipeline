import abc, contextlib, dataclassy, numbers, numpy as np, pathlib, shutil, tempfile, unittest, more_itertools

from astropath.utilities import units
from astropath.utilities.tableio import readtable

def assertAlmostEqual(a, b, **kwargs):
  if isinstance(a, np.ndarray) and not a.shape: a = a[()]
  if isinstance(b, np.ndarray) and not b.shape: b = b[()]
  if isinstance(a, units.safe.Distance):
    return units.np.testing.assert_allclose(a, b, **kwargs)
  elif isinstance(a, numbers.Number):
    if isinstance(b, units.safe.Distance): b = float(b)
    return np.testing.assert_allclose(a, b, **kwargs)
  elif dataclassy.functions.is_dataclass(type(a)) and type(a) == type(b):
    try:
      for field in dataclassy.fields(type(a)):
        assertAlmostEqual(getattr(a, field), getattr(b, field), **kwargs)
    except AssertionError:
      np.testing.assert_equal(a, b)
  elif isinstance(a, np.ndarray):
    units.np.testing.assert_allclose(a, b, **kwargs)
  else:
    return np.testing.assert_equal(a, b)

#compare two .csv files with the given paths and holding lines of the given datatype 
def compare_two_csv_files(filedir,reffiledir,filename,dataclass,checkorder=True,checknewlines=True,rtol=1e-5) :
  rows = readtable(filedir/filename, dataclass, checkorder=checkorder, checknewlines=checknewlines)
  targetrows = readtable(reffiledir/filename, dataclass, checkorder=checkorder, checknewlines=checknewlines)
  for row, target in more_itertools.zip_equal(rows, targetrows):
    assertAlmostEqual(row, target, rtol=rtol)

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
    super().setUpClass()

  @property
  @abc.abstractmethod
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
    super().tearDownClass()

class TestBaseCopyInput(abc.ABC, unittest.TestCase):
  @classmethod
  def removecopiedinput(cls): return True

  @classmethod
  @abc.abstractmethod
  def filestocopy(cls): pass

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    for copyfrom, copytofolder in cls.filestocopy():
      if isinstance(copytofolder, tuple):
        copytofolder, copyto = copytofolder
        copyto = copytofolder/copyto
      else:
        copyto = copytofolder
      copytofolder.mkdir(exist_ok=True, parents=True)
      shutil.copy(copyfrom, copyto)

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    if cls.removecopiedinput():
      for copyfrom, copytofolder in cls.filestocopy():
        if isinstance(copytofolder, tuple):
          folder, name = copytofolder
          copyto = folder/name
        else:
          copyto = copytofolder/copyfrom.name
        try:
          copyto.unlink()
        except FileNotFoundError:
          pass
