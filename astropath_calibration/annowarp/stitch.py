import abc, cvxpy as cp, dataclasses

from ..utilities import units
from ..utilities.tableio import writetable
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

class AnnoWarpStitchResultBase(abc.ABC):
  def __init__(self, *, imscale, **kwargs):
    self.imscale = imscale
    super().__init__(**kwargs)

  @abc.abstractmethod
  def dxvec(self, alignmentresult):
    pass

  def residual(self, alignmentresult):
    return alignmentresult.dxvec - self.dxvec(alignmentresult)

  def writestitchresult(self, *, filename, **kwargs):
    writetable(filename, self.stitchresultentries, **kwargs)

  @abc.abstractproperty
  def stitchresultentries(self): pass

class AnnoWarpStitchResultNoCvxpyBase(AnnoWarpStitchResultBase):
  pass

class AnnoWarpStitchResultCvxpyBase(AnnoWarpStitchResultBase):
  def __init__(self, *, problem, **kwargs):
    self.problem = problem
    super().__init__(**kwargs)

  def residual(self, alignmentresult):
    return units.nominal_values(super().residual(alignmentresult))

  @classmethod
  @abc.abstractmethod
  def makecvxpyvariables(cls): return {}

  @classmethod
  @abc.abstractmethod
  def cvxpydxvec(cls, alignmentresult, **cvxpyvariables): pass

  @classmethod
  def cvxpyresidual(cls, alignmentresult, **cvxpyvariables):
    return units.nominal_values(alignmentresult.dxvec)/alignmentresult.onepixel - cls.cvxpydxvec(alignmentresult, **cvxpyvariables)

class AnnoWarpStitchResultDefaultModelBase(AnnoWarpStitchResultBase):
  def __init__(self, *, coeffrelativetobigtile, bigtileindexcoeff, constant, **kwargs):
    self.coeffrelativetobigtile = coeffrelativetobigtile
    self.bigtileindexcoeff = bigtileindexcoeff
    self.constant = constant
    super().__init__(**kwargs)

  def dxvec(self, alignmentresult):
    return (
      self.coeffrelativetobigtile @ alignmentresult.centerrelativetobigtile
      + self.bigtileindexcoeff @ alignmentresult.bigtileindex
      + self.constant
    )

  @property
  def stitchresultentries(self):
    return (
      AnnoWarpStitchResultEntry(
        n=1,
        value=self.coeffrelativetobigtile[0,0],
        description="coefficient of delta x as a function of x within the tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=2,
        value=self.coeffrelativetobigtile[0,1],
        description="coefficient of delta x as a function of y within the tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=3,
        value=self.coeffrelativetobigtile[1,0],
        description="coefficient of delta y as a function of x within the tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=4,
        value=self.coeffrelativetobigtile[1,1],
        description="coefficient of delta y as a function of x within the tile",
        pscale=self.imscale,
      ),

      AnnoWarpStitchResultEntry(
        n=5,
        value=self.bigtileindexcoeff[0,0],
        description="coefficient of delta x as a function of tile index in x",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=6,
        value=self.bigtileindexcoeff[0,1],
        description="coefficient of delta x as a function of tile index in y",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=7,
        value=self.bigtileindexcoeff[1,0],
        description="coefficient of delta y as a function of tile index in x",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=8,
        value=self.bigtileindexcoeff[1,1],
        description="coefficient of delta y as a function of tile index in x",
        pscale=self.imscale,
      ),

      AnnoWarpStitchResultEntry(
        n=9,
        value=self.constant[0],
        description="constant piece in delta x",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=10,
        value=self.constant[1],
        description="constant piece in delta y",
        pscale=self.imscale,
      ),
    )

class AnnoWarpStitchResultDefaultModel(AnnoWarpStitchResultDefaultModelBase, AnnoWarpStitchResultNoCvxpyBase):
  pass

class AnnoWarpStitchResultDefaultModelCvxpy(AnnoWarpStitchResultDefaultModelBase, AnnoWarpStitchResultCvxpyBase):
  def __init__(self, *, coeffrelativetobigtile, bigtileindexcoeff, constant, imscale, **kwargs):
    onepixel = self.onepixel = units.Distance(pixels=1, pscale=imscale)
    super().__init__(
      coeffrelativetobigtile=coeffrelativetobigtile.value,
      bigtileindexcoeff=bigtileindexcoeff.value * onepixel,
      constant=constant.value * onepixel,
      imscale=imscale,
      **kwargs,
    )
    self.coeffrelativetobigtilevar = coeffrelativetobigtile
    self.bigtileindexcoeffvar = bigtileindexcoeff
    self.constantvar = constant

  @classmethod
  def makecvxpyvariables(cls): return {
    "coeffrelativetobigtile": cp.Variable(shape=(2, 2)),
    "bigtileindexcoeff": cp.Variable(shape=(2, 2)),
    "constant": cp.Variable(shape=2),
  }

  @classmethod
  def cvxpydxvec(cls, alignmentresult, *, coeffrelativetobigtile, bigtileindexcoeff, constant):
    return (
      coeffrelativetobigtile @ (alignmentresult.centerrelativetobigtile / alignmentresult.onepixel)
      + bigtileindexcoeff @ alignmentresult.bigtileindex
      + constant
    )

class AnnoWarpStitchResultTwoTilesBase(AnnoWarpStitchResultBase):
  def __init__(self, *, coeffrelativetoevenbigtile, coeffrelativetooddbigtile, evenbigtileindexcoeff, oddbigtileindexcoeff, constant, **kwargs):
    self.coeffrelativetoevenbigtile = coeffrelativetoevenbigtile
    self.coeffrelativetooddbigtile = coeffrelativetooddbigtile
    self.evenbigtileindexcoeff = evenbigtileindexcoeff
    self.oddbigtileindexcoeff = oddbigtileindexcoeff
    self.constant = constant
    super().__init__(**kwargs)

  def dxvec(self, alignmentresult):
    return (
      self.coeffrelativetoevenbigtile @ alignmentresult.centerrelativetoevenbigtile
      + self.coeffrelativetooddbigtile @ alignmentresult.centerrelativetooddbigtile
      + self.evenbigtileindexcoeff @ alignmentresult.evenbigtileindex
      + self.oddbigtileindexcoeff @ alignmentresult.oddbigtileindex
      + self.constant
    )

  @property
  def stitchresultentries(self):
    return (
      AnnoWarpStitchResultEntry(
        n=1,
        value=self.coeffrelativetoevenbigtile[0,0],
        description="coefficient of delta x as a function of x within the even tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=2,
        value=self.coeffrelativetoevenbigtile[0,1],
        description="coefficient of delta x as a function of y within the even tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=3,
        value=self.coeffrelativetoevenbigtile[1,0],
        description="coefficient of delta y as a function of x within the even tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=4,
        value=self.coeffrelativetoevenbigtile[1,1],
        description="coefficient of delta y as a function of x within the even tile",
        pscale=self.imscale,
      ),

      AnnoWarpStitchResultEntry(
        n=5,
        value=self.coeffrelativetooddbigtile[0,0],
        description="coefficient of delta x as a function of x within the odd tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=6,
        value=self.coeffrelativetooddbigtile[0,1],
        description="coefficient of delta x as a function of y within the odd tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=7,
        value=self.coeffrelativetooddbigtile[1,0],
        description="coefficient of delta y as a function of x within the odd tile",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=8,
        value=self.coeffrelativetooddbigtile[1,1],
        description="coefficient of delta y as a function of x within the odd tile",
        pscale=self.imscale,
      ),

      AnnoWarpStitchResultEntry(
        n=9,
        value=self.evenbigtileindexcoeff[0,0],
        description="coefficient of delta x as a function of even tile index in x",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=10,
        value=self.evenbigtileindexcoeff[0,1],
        description="coefficient of delta x as a function of even tile index in y",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=11,
        value=self.evenbigtileindexcoeff[1,0],
        description="coefficient of delta y as a function of even tile index in x",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=12,
        value=self.evenbigtileindexcoeff[1,1],
        description="coefficient of delta y as a function of even tile index in x",
        pscale=self.imscale,
      ),

      AnnoWarpStitchResultEntry(
        n=13,
        value=self.oddbigtileindexcoeff[0,0],
        description="coefficient of delta x as a function of odd tile index in x",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=14,
        value=self.oddbigtileindexcoeff[0,1],
        description="coefficient of delta x as a function of odd tile index in y",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=15,
        value=self.oddbigtileindexcoeff[1,0],
        description="coefficient of delta y as a function of odd tile index in x",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=16,
        value=self.oddbigtileindexcoeff[1,1],
        description="coefficient of delta y as a function of odd tile index in x",
        pscale=self.imscale,
      ),

      AnnoWarpStitchResultEntry(
        n=17,
        value=self.constant[0],
        description="constant piece in delta x",
        pscale=self.imscale,
      ),
      AnnoWarpStitchResultEntry(
        n=18,
        value=self.constant[1],
        description="constant piece in delta y",
        pscale=self.imscale,
      ),
    )

class AnnoWarpStitchResultTwoTiles(AnnoWarpStitchResultTwoTilesBase, AnnoWarpStitchResultNoCvxpyBase):
  pass

class AnnoWarpStitchResultTwoTilesCvxpy(AnnoWarpStitchResultTwoTilesBase, AnnoWarpStitchResultCvxpyBase):
  def __init__(self, *, coeffrelativetoevenbigtile, coeffrelativetooddbigtile, evenbigtileindexcoeff, oddbigtileindexcoeff, constant, imscale, **kwargs):
    onepixel = self.onepixel = units.Distance(pixels=1, pscale=imscale)
    super().__init__(
      coeffrelativetoevenbigtile=coeffrelativetoevenbigtile.value,
      coeffrelativetooddbigtile=coeffrelativetooddbigtile.value,
      evenbigtileindexcoeff=evenbigtileindexcoeff.value * onepixel,
      oddbigtileindexcoeff=oddbigtileindexcoeff.value * onepixel,
      constant=constant.value * onepixel,
      imscale=imscale,
      **kwargs,
    )
    self.coeffrelativetoevenbigtilevar = coeffrelativetoevenbigtile
    self.coeffrelativetooddbigtilevar = coeffrelativetooddbigtile
    self.evenbigtileindexcoeffvar = evenbigtileindexcoeff
    self.oddbigtileindexcoeffvar = oddbigtileindexcoeff
    self.constantvar = constant

  @classmethod
  def makecvxpyvariables(cls): return {
    "coeffrelativetoevenbigtile": cp.Variable(shape=(2, 2)),
    "coeffrelativetooddbigtile": cp.Variable(shape=(2, 2)),
    "evenbigtileindexcoeff": cp.Variable(shape=(2, 2)),
    "oddbigtileindexcoeff": cp.Variable(shape=(2, 2)),
    "constant": cp.Variable(shape=2),
  }

  @classmethod
  def cvxpydxvec(cls, alignmentresult, *, coeffrelativetoevenbigtile, coeffrelativetooddbigtile, evenbigtileindexcoeff, oddbigtileindexcoeff, constant):
    return (
      coeffrelativetoevenbigtile @ (alignmentresult.centerrelativetoevenbigtile / alignmentresult.onepixel)
      + coeffrelativetooddbigtile @ (alignmentresult.centerrelativetooddbigtile / alignmentresult.onepixel)
      + evenbigtileindexcoeff @ alignmentresult.evenbigtileindex
      + oddbigtileindexcoeff @ alignmentresult.oddbigtileindex
      + constant
    )

@dataclasses.dataclass
class AnnoWarpStitchResultEntry(DataClassWithDistances):
  pixelsormicrons = "pixels"
  def __powerfordescription(self):
    return {
      "coefficient of delta x as a function of x within the tile": 0,
      "coefficient of delta x as a function of y within the tile": 0,
      "coefficient of delta y as a function of x within the tile": 0,
      "coefficient of delta y as a function of y within the tile": 0,
      "coefficient of delta x as a function of tile index in x": 1,
      "coefficient of delta x as a function of tile index in y": 1,
      "coefficient of delta y as a function of tile index in x": 1,
      "coefficient of delta y as a function of tile index in y": 1,
      "constant piece in delta x": 1,
      "constant piece in delta y": 1,
    }[self.description]
  n: int
  value: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, power=__powerfordescription)
  description: str
  pscale: dataclasses.InitVar[float] = None
  readingfromfile: dataclasses.InitVar[bool] = False
