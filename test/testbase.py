import abc, contextlib, dataclassy, more_itertools, numbers, numpy as np, pathlib, PIL.Image, re, shutil, tempfile, unittest

from astropath.utilities import units
from astropath.utilities.miscfileio import rm_missing_ok
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
def compare_two_csv_files(filedir,reffiledir,filename,dataclass,checkorder=True,checknewlines=True,rtol=1e-5,extrakwargs={}) :
  rows = readtable(filedir/filename, dataclass, checkorder=checkorder, checknewlines=checknewlines, extrakwargs=extrakwargs)
  targetrows = readtable(reffiledir/filename, dataclass, checkorder=checkorder, checknewlines=checknewlines, extrakwargs=extrakwargs)
  for row, target in more_itertools.zip_equal(rows, targetrows):
    assertAlmostEqual(row, target, rtol=rtol)

def compare_two_images(image, ref):
  with PIL.Image.open(image) as im, PIL.Image.open(ref) as refim:
    np.testing.assert_array_equal(np.asarray(im), np.asarray(refim))

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

class TestBase(abc.ABC, unittest.TestCase):
  def setUp(self):
    self.maxDiff = None
    super().setUp()

  def assertLengthEqual(self, sequence, length, msg=None):
    """Fail the test unless the sequence is the desired length."""
    if len(sequence) != length:
      standardMsg = "length of %r is %d, not %d" % (sequence, len(sequence), length)
      # _formatMessage ensures the longMessage option is respected
      msg = self._formatMessage(msg, standardMsg)
      raise self.failureException(msg)

class TestBaseSaveOutput(TestBase):
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
      rm_missing_ok(filename)

  def setUp(self):
    super().setUp()
    self.removeoutput()

  def tearDown(self):
    super().tearDown()

  @classmethod
  def tearDownClass(cls):
    cls.__output.__exit__(None, None, None)
    super().tearDownClass()

class TestBaseCopyInput(TestBase):
  class FileToCopy:
    def __init__(self, copyfrom, copytofolder, copyto=None, refind=None, rereplace=None, *, removecopiedinput):
      if copyto is None:
        copyto = copytofolder/copyfrom.name
      else:
        copyto = copytofolder/copyto

      self.copyfrom = copyfrom
      self.copytofolder = copytofolder
      self.copyto = copyto
      self.refind = refind
      self.rereplace = rereplace
      if (refind is not None) != (rereplace is not None):
        raise ValueError("Have to provide both refind and rereplace or neither")
      self.removecopiedinput = removecopiedinput

    def __enter__(self):
      self.copytofolder.mkdir(exist_ok=True, parents=True)
      if self.refind is self.rereplace is None:
        shutil.copy(self.copyfrom, self.copyto)
      else:
        with open(self.copyfrom) as f, open(self.copyto, "w") as newf:
          newf.write(re.sub(self.refind, self.rereplace, f.read()))
    def __exit__(self, *exc):
      if self.removecopiedinput:
        rm_missing_ok(self.copyto)

  @classmethod
  def removecopiedinput(cls): return True

  @classmethod
  @abc.abstractmethod
  def filestocopy(cls): pass

  @classmethod
  def setUpClass(cls):
    cls.__input = contextlib.ExitStack()
    cls.__input.__enter__()
    super().setUpClass()
    for filetocopyargs in cls.filestocopy():
      cls.__input.enter_context(cls.FileToCopy(*filetocopyargs, removecopiedinput=cls.removecopiedinput()))

  @classmethod
  def tearDownClass(cls):
    try:
      cls.__input.close()
    finally:
      super().tearDownClass()
