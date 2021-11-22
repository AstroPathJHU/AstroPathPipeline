class OptionalImport:
  """
  Helper class to import optional modules when needed,
  but allow doing other things even without that import
  """
  def __init__(self, moduletoimport, pipinstall=None):
    self.__module = None
    self.moduletoimport = moduletoimport
    if pipinstall is None: pipinstall = moduletoimport
    self.pipinstall = pipinstall

  def doimport(self):
    if self.__module is not None: return
    try:
      self.__module = __import__(self.moduletoimport)
    except ImportError:
      raise ImportError(f"Please pip install {self.pipinstall} to use this feature")
    self.initmodule()
    self.doimport = lambda self: None

  def initmodule(self):
    pass

  def __getattr__(self, attr):
    self.doimport()
    return getattr(self.__module, attr)

  @property
  def module(self):
    if self.module is None: return self
    return self.__module

class OgrImport(OptionalImport):
  def __init__(self):
    super().__init__("osgeo.ogr", "gdal")
  def initmodule(self):
    super().initmodule()
    self.UseExceptions()

cvxpy = OptionalImport("cvxpy")
ogr = OgrImport()
pyopencl = OptionalImport("pyopencl")
pyvips = OptionalImport("pyvips")
reikna = OptionalImport("reikna")
