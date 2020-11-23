import abc, cvxpy as cp, dataclasses, numpy as np

from ..utilities import units
from ..utilities.tableio import writetable
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

class ThingWithImscale(abc.ABC):
  @abc.abstractproperty
  def imscale(self): pass
  @property
  def oneimpixel(self): return units.onepixel(pscale=self.imscale)
  @property
  def oneimmicron(self): return units.onemicron(pscale=self.imscale)

class AnnoWarpStitchResultBase(ThingWithImscale):
  def __init__(self, *, imscale, **kwargs):
    self.__imscale = imscale
    super().__init__(**kwargs)

  @property
  def imscale(self): return self.__imscale

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
  def __init__(self, *, A, b, c, flatresult, **kwargs):
    self.A = A
    self.b = b
    self.c = c
    self.flatresult = flatresult
    super().__init__(**kwargs)

  @classmethod
  @abc.abstractmethod
  def nparams(cls): pass

  @classmethod
  @abc.abstractmethod
  def Abccontributions(cls, alignmentresult): pass

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
  def __init__(self, flatresult, **kwargs):
    coeffrelativetobigtile, bigtileindexcoeff, constant = np.split(flatresult, [4, 8])
    coeffrelativetobigtile = coeffrelativetobigtile.reshape(2, 2)
    bigtileindexcoeff = bigtileindexcoeff.reshape(2, 2)
    super().__init__(flatresult=flatresult, coeffrelativetobigtile=coeffrelativetobigtile, bigtileindexcoeff=bigtileindexcoeff, constant=constant, **kwargs)

  @classmethod
  def nparams(cls): return 10

  @classmethod
  def Abccontributions(cls, alignmentresult):
    nparams = cls.nparams()
    (
      crtbt_xx,
      crtbt_xy,
      crtbt_yx,
      crtbt_yy,
      bti_xx,
      bti_xy,
      bti_yx,
      bti_yy,
      const_x,
      const_y,
    ) = range(nparams)
    A = np.zeros(shape=(nparams, nparams), dtype=units.unitdtype)
    b = np.zeros(shape=nparams, dtype=units.unitdtype)
    c = 0

    crtbt = alignmentresult.centerrelativetobigtile
    bti = alignmentresult.bigtileindex

    dxvec = units.nominal_values(alignmentresult.dxvec)
    invcov = units.np.linalg.inv(alignmentresult.covariance)

    A[crtbt_xx:crtbt_xy+1, crtbt_xx:crtbt_xy+1] += np.outer(crtbt, crtbt) * invcov[0,0]
    A[crtbt_yx:crtbt_yy+1, crtbt_xx:crtbt_xy+1] += np.outer(crtbt, crtbt) * invcov[0,1]
    A[crtbt_xx:crtbt_xy+1, crtbt_yx:crtbt_yy+1] += np.outer(crtbt, crtbt) * invcov[1,0]
    A[crtbt_yx:crtbt_yy+1, crtbt_yx:crtbt_yy+1] += np.outer(crtbt, crtbt) * invcov[1,1]

    A[crtbt_xx:crtbt_xy+1, bti_xx:bti_xy+1] += np.outer(crtbt, bti) * invcov[0,0]
    A[crtbt_yx:crtbt_yy+1, bti_xx:bti_xy+1] += np.outer(crtbt, bti) * invcov[0,1]
    A[crtbt_xx:crtbt_xy+1, bti_yx:bti_yy+1] += np.outer(crtbt, bti) * invcov[1,0]
    A[crtbt_yx:crtbt_yy+1, bti_yx:bti_yy+1] += np.outer(crtbt, bti) * invcov[1,1]

    A[crtbt_xx:crtbt_xy+1, const_x] += crtbt * invcov[0,0]
    A[crtbt_xx:crtbt_xy+1, const_y] += crtbt * invcov[0,1]
    A[crtbt_yx:crtbt_yy+1, const_x] += crtbt * invcov[1,0]
    A[crtbt_yx:crtbt_yy+1, const_y] += crtbt * invcov[1,1]

    A[bti_xx:bti_xy+1, crtbt_xx:crtbt_xy+1] += np.outer(bti, crtbt) * invcov[0,0]
    A[bti_yx:bti_yy+1, crtbt_xx:crtbt_xy+1] += np.outer(bti, crtbt) * invcov[0,1]
    A[bti_xx:bti_xy+1, crtbt_yx:crtbt_yy+1] += np.outer(bti, crtbt) * invcov[1,0]
    A[bti_yx:bti_yy+1, crtbt_yx:crtbt_yy+1] += np.outer(bti, crtbt) * invcov[1,1]

    A[bti_xx:bti_xy+1, bti_xx:bti_xy+1] += np.outer(bti, bti) * invcov[0,0]
    A[bti_yx:bti_yy+1, bti_xx:bti_xy+1] += np.outer(bti, bti) * invcov[0,1]
    A[bti_xx:bti_xy+1, bti_yx:bti_yy+1] += np.outer(bti, bti) * invcov[1,0]
    A[bti_yx:bti_yy+1, bti_yx:bti_yy+1] += np.outer(bti, bti) * invcov[1,1]

    A[bti_xx:bti_xy+1, const_x] += bti * invcov[0,0]
    A[bti_xx:bti_xy+1, const_y] += bti * invcov[0,1]
    A[bti_yx:bti_yy+1, const_x] += bti * invcov[1,0]
    A[bti_yx:bti_yy+1, const_y] += bti * invcov[1,1]

    A[const_x, crtbt_xx:crtbt_xy+1] += crtbt * invcov[0,0]
    A[const_y, crtbt_xx:crtbt_xy+1] += crtbt * invcov[0,1]
    A[const_x, crtbt_yx:crtbt_yy+1] += crtbt * invcov[1,0]
    A[const_y, crtbt_yx:crtbt_yy+1] += crtbt * invcov[1,1]

    A[const_x, bti_xx:bti_xy+1] += bti * invcov[0,0]
    A[const_y, bti_xx:bti_xy+1] += bti * invcov[0,1]
    A[const_x, bti_yx:bti_yy+1] += bti * invcov[1,0]
    A[const_y, bti_yx:bti_yy+1] += bti * invcov[1,1]

    A[const_x, const_x] += invcov[0,0]
    A[const_x, const_y] += invcov[0,1]
    A[const_y, const_x] += invcov[1,0]
    A[const_y, const_y] += invcov[1,1]

    b[crtbt_xx:crtbt_xy+1] -= 2 * crtbt * invcov[0,0] * dxvec[0]
    b[crtbt_xx:crtbt_xy+1] -= 2 * crtbt * invcov[0,1] * dxvec[1]
    b[crtbt_yx:crtbt_yy+1] -= 2 * crtbt * invcov[1,0] * dxvec[0]
    b[crtbt_yx:crtbt_yy+1] -= 2 * crtbt * invcov[1,1] * dxvec[1]

    b[bti_xx:bti_xy+1] -= 2 * bti * invcov[0,0] * dxvec[0]
    b[bti_xx:bti_xy+1] -= 2 * bti * invcov[0,1] * dxvec[1]
    b[bti_yx:bti_yy+1] -= 2 * bti * invcov[1,0] * dxvec[0]
    b[bti_yx:bti_yy+1] -= 2 * bti * invcov[1,1] * dxvec[1]

    b[const_x] -= 2 * invcov[0,0] * dxvec[0]
    b[const_x] -= 2 * invcov[0,1] * dxvec[1]
    b[const_y] -= 2 * invcov[1,0] * dxvec[0]
    b[const_y] -= 2 * invcov[1,1] * dxvec[1]

    c += dxvec @ invcov @ dxvec

    return A, b, c

class AnnoWarpStitchResultDefaultModelCvxpy(AnnoWarpStitchResultDefaultModelBase, AnnoWarpStitchResultCvxpyBase):
  def __init__(self, *, coeffrelativetobigtile, bigtileindexcoeff, constant, imscale, **kwargs):
    onepixel = units.onepixel(pscale=imscale)
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
