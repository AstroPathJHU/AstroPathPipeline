import importlib

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
      self.__module = importlib.import_module(self.moduletoimport)
    except ImportError:
      raise ImportError(f"Please pip install {self.pipinstall} to use this feature")
    self.initmodule()
    self.doimport = lambda: None

  def initmodule(self):
    pass

  def __getattr__(self, attr):
    self.doimport()
    return getattr(self.__module, attr)

class OgrImport(OptionalImport):
  def __init__(self):
    super().__init__("osgeo.ogr", "gdal")
  def initmodule(self):
    super().initmodule()
    self.UseExceptions()

class NNUNetImport(OptionalImport) :
  def __init__(self):
    super().__init__("nnunet")
  def initmodule(self):
    super().initmodule()
    import nnunet.inference.predict
    import nnunet.paths
    self.inference.predict = nnunet.inference.predict
    self.paths = nnunet.paths

cvxpy = OptionalImport("cvxpy")
deepcell = OptionalImport("deepcell")
nnunet = NNUNetImport()
ogr = OgrImport()
pyopencl = OptionalImport("pyopencl")
pyvips = OptionalImport("pyvips")
reikna = OptionalImport("reikna")
