import abc, collections, cvxpy as cp, itertools, more_itertools, numpy as np, re

from ..utilities import units
from ..utilities.misc import dict_zip_equal
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
    nominal = list(self.stitchresultnominalentries)
    for entry, power in more_itertools.zip_equal(nominal, self.variablepowers()):
      if entry.powerfordescription(entry) != power:
        raise ValueError(f"Wrong power for {entry.description!r}: expected {power}, got {entry.powerfordescription(entry)}")
    return list(itertools.chain(nominal, self.stitchresultcovarianceentries))

  @classmethod
  @abc.abstractmethod
  def variablepowers(self): pass

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

  @classmethod
  def constraintAbccontributions(cls, mus, sigmas):
    if mus is sigmas is None: return 0, 0, 0
    nparams = cls.nparams()
    A = np.zeros(shape=(nparams, nparams), dtype=units.unitdtype)
    b = np.zeros(shape=nparams, dtype=units.unitdtype)
    c = 0
    for i, mu, sigma in more_itertools.zip_equal(range(nparams), mus, sigmas):
      if mu is sigma is None: continue
      A[i,i] += 1/sigma**2
      b[i] -= 2*mu/sigma**2
      c += (mu/sigma)**2
    return A, b, c

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

  @classmethod
  def constraintquadforms(cls, cvxpyvariables, mus, sigmas, *, imscale):
    if mus is sigmas is None: return 0
    onepixel = units.onepixel(imscale)
    result = 0
    musdict = {}
    sigmasdict = {}
    iterator = iter(more_itertools.zip_equal(mus, sigmas, cls.variablepowers(), range(sum(v.size for k, v in cvxpyvariables.items()))))
    for name, variable in cvxpyvariables.items():
      musdict[name] = np.zeros(shape=variable.shape)
      sigmasdict[name] = np.zeros(shape=variable.shape)
      raveledmu = musdict[name].ravel()
      raveledsigma = sigmasdict[name].ravel()
      for i, (mu, sigma, power, _) in enumerate(itertools.islice(iterator, variable.size)):
        if mu is sigma is None:
          raveledmu[i] = 0
          raveledsigma[i] = float("inf")
        else:
          raveledmu[i] = mu / onepixel**power
          raveledsigma[i] = sigma / onepixel**power

    with np.testing.assert_raises(StopIteration):
      next(iterator)

    for k, (variable, mu, sigma) in dict_zip_equal(cvxpyvariables, musdict, sigmasdict).items():
      result += cp.sum(((variable-mu)/sigma)**2)

    return result

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

  @classmethod
  def variablepowers(cls):
    return 0, 0, 0, 0, 1, 1, 1, 1, 1, 1

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
  @classmethod
  def powerfordescription(cls, selfordescription):
    if isinstance(selfordescription, cls):
      description = selfordescription.description
    else:
      description = selfordescription
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
    covmatch = re.match(r"covariance\((.*), (.*)\)", description)
    if covmatch:
      return dct[covmatch.group(1)] + dct[covmatch.group(2)]
    else:
      return dct[description]
  n: int
  value: distancefield(pixelsormicrons=pixelsormicrons, power=lambda self: self.powerfordescription(self))
  description: str
