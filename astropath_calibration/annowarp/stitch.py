import abc, collections, cvxpy as cp, itertools, numpy as np, re

from ..utilities import units
from ..utilities.tableio import writetable
from ..utilities.units.dataclasses import DataClassWithPscale, distancefield

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
  def dxvec(self, qptiffcoordinate, *, apscale):
    pass

  def residual(self, alignmentresult, *, apscale):
    return alignmentresult.dxvec - self.dxvec(alignmentresult, apscale=apscale)

  def writestitchresult(self, *, filename, **kwargs):
    writetable(filename, self.allstitchresultentries, **kwargs)

  EntryLite = collections.namedtuple("EntryLite", "value description")

  @abc.abstractproperty
  def stitchresultentries(self): pass

  @property
  def stitchresultnominalentries(self):
    for n, (value, description) in enumerate(self.stitchresultentries, start=1):
      yield AnnoWarpStitchResultEntry(
        n=n,
        value=units.nominal_value(value),
        description=description,
        pscale=self.imscale,
      )

  @abc.abstractproperty
  def stitchresultcovarianceentries(self): pass

  @property
  def allstitchresultentries(self):
    return list(itertools.chain(self.stitchresultnominalentries, self.stitchresultcovarianceentries))

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

  @property
  def stitchresultcovarianceentries(self):
    entries = self.stitchresultentries
    for n, ((value1, description1), (value2, description2)) in enumerate(itertools.combinations_with_replacement(entries, 2), start=len(entries)+1):
      yield AnnoWarpStitchResultEntry(
        n=n,
        value=np.array(units.covariance_matrix([value1, value2]))[0, 1],
        description="covariance("+description1+", "+description2+")",
        pscale=self.imscale,
      )

class AnnoWarpStitchResultCvxpyBase(AnnoWarpStitchResultBase):
  def __init__(self, *, problem, **kwargs):
    self.problem = problem
    super().__init__(**kwargs)

  def residual(self, *args, **kwargs):
    return units.nominal_values(super().residual(*args, **kwargs))

  @classmethod
  @abc.abstractmethod
  def makecvxpyvariables(cls): return {}

  @classmethod
  @abc.abstractmethod
  def cvxpydxvec(cls, alignmentresult, **cvxpyvariables): pass

  @classmethod
  def cvxpyresidual(cls, alignmentresult, **cvxpyvariables):
    return units.nominal_values(alignmentresult.dxvec)/alignmentresult.onepixel - cls.cvxpydxvec(alignmentresult, **cvxpyvariables)

  @property
  def stitchresultcovarianceentries(self): return []

class AnnoWarpStitchResultDefaultModelBase(AnnoWarpStitchResultBase):
  def __init__(self, *, coeffrelativetobigtile, bigtileindexcoeff, constant, **kwargs):
    self.coeffrelativetobigtile = coeffrelativetobigtile
    self.bigtileindexcoeff = bigtileindexcoeff
    self.constant = constant
    super().__init__(**kwargs)

  def dxvec(self, qptiffcoordinate, *, apscale):
    coeffrelativetobigtile = self.coeffrelativetobigtile
    bigtileindexcoeff = units.convertpscale(self.bigtileindexcoeff, self.imscale, apscale)
    constant = units.convertpscale(self.constant, self.imscale, apscale)
    return (
      coeffrelativetobigtile @ qptiffcoordinate.centerrelativetobigtile
      + bigtileindexcoeff @ qptiffcoordinate.bigtileindex
      + constant
    )

  @property
  def stitchresultentries(self):
    return (
      self.EntryLite(
        value=self.coeffrelativetobigtile[0,0],
        description="coefficient of delta x as a function of x within the tile",
      ),
      self.EntryLite(
        value=self.coeffrelativetobigtile[0,1],
        description="coefficient of delta x as a function of y within the tile",
      ),
      self.EntryLite(
        value=self.coeffrelativetobigtile[1,0],
        description="coefficient of delta y as a function of x within the tile",
      ),
      self.EntryLite(
        value=self.coeffrelativetobigtile[1,1],
        description="coefficient of delta y as a function of y within the tile",
      ),

      self.EntryLite(
        value=self.bigtileindexcoeff[0,0],
        description="coefficient of delta x as a function of tile index in x",
      ),
      self.EntryLite(
        value=self.bigtileindexcoeff[0,1],
        description="coefficient of delta x as a function of tile index in y",
      ),
      self.EntryLite(
        value=self.bigtileindexcoeff[1,0],
        description="coefficient of delta y as a function of tile index in x",
      ),
      self.EntryLite(
        value=self.bigtileindexcoeff[1,1],
        description="coefficient of delta y as a function of tile index in y",
      ),

      self.EntryLite(
        value=self.constant[0],
        description="constant piece in delta x",
      ),
      self.EntryLite(
        value=self.constant[1],
        description="constant piece in delta y",
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

class AnnoWarpStitchResultEntry(DataClassWithPscale):
  pixelsormicrons = "pixels"
  def __powerfordescription(self):
    dct = {
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
    }
    covmatch = re.match(r"covariance\((.*), (.*)\)", self.description)
    if covmatch:
      return dct[covmatch.group(1)] + dct[covmatch.group(2)]
    else:
      return dct[self.description]
  n: int
  value: distancefield(pixelsormicrons=pixelsormicrons, power=__powerfordescription)
  description: str
