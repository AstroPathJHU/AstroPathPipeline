import abc, collections, itertools, more_itertools, numpy as np, re

from ...utilities import units
from ...utilities.misc import dict_zip_equal
from ...utilities.optionalimports import cvxpy as cp
from ...utilities.tableio import writetable
from ...utilities.units.dataclasses import DataClassWithImscale, distancefield

class AnnoWarpStitchResultBase(units.ThingWithImscale):
  """
  Base class for annowarp stitch results.

  Stitch result classes basically have 2 degrees of freedom:
   1. the stitching model to use
   2. how to solve the equation (with cvxpy or standalone linear algebra)
  """
  def __init__(self, *, pscale, apscale, constraintmus=None, constraintsigmas=None, **kwargs):
    self.__pscale = pscale
    self.__apscale = apscale
    self.constraintmus = constraintmus
    self.constraintsigmas = constraintsigmas
    super().__init__(**kwargs)

  @property
  def pscale(self): return self.__pscale
  @property
  def apscale(self): return self.__apscale

  @abc.abstractmethod
  def dxvec(self, qptiffcoordinate, *, apscale):
    r"""
    \Delta\vec{x} for the qptiff coordinate, calculated from the
    fitted stitching parameters
    """

  def residual(self, alignmentresult, *, apscale):
    r"""
    The residual, \Delta\vec{x} for an alignment result - \Delta\vec{x} predicted
    for that alignment from the stitching model
    """
    return alignmentresult.dxvec - self.dxvec(alignmentresult, apscale=apscale)

  def writestitchresult(self, *, filename, **kwargs):
    """
    Write the fitted parameters to a csv file
    """
    writetable(filename, self.allstitchresultentries, **kwargs)

  EntryLite = collections.namedtuple("EntryLite", "value description")

  @property
  @abc.abstractmethod
  def stitchresultentries(self):
    """
    A list of EntryLite objects that give the stitching parameters.
    """

  @property
  def stitchresultnominalentries(self):
    """
    AnnoWarpStitchResultEntries for the fitted values of the parameters
    """
    for n, (value, description) in enumerate(self.stitchresultentries, start=1):
      yield AnnoWarpStitchResultEntry(
        n=n,
        value=units.nominal_value(value),
        description=description,
        pscale=self.pscale,
        apscale=self.apscale,
      )

  @property
  def stitchresultcovarianceentries(self):
    """
    AnnoWarpStitchResultEntries for the parameter covariance matrix
    """
    entries = self.stitchresultentries
    if all(units.std_dev(value) == 0 for value, description in entries): return
    for n, ((value1, description1), (value2, description2)) in enumerate(itertools.combinations_with_replacement(entries, 2), start=len(entries)+1):
      yield AnnoWarpStitchResultEntry(
        n=n,
        value=np.array(units.covariance_matrix([value1, value2]))[0, 1],
        description="cov("+description1+", "+description2+")",
        pscale=self.pscale,
        apscale=self.apscale,
      )

  @property
  def stitchresultconstraintentries(self):
    entries = self.stitchresultentries
    nentries = len(entries)
    mus = self.constraintmus
    sigmas = self.constraintsigmas
    if mus is sigmas is None: return
    for n, ((value, description), mu, sigma) in enumerate(more_itertools.zip_equal(entries, mus, sigmas), start=nentries + nentries*(nentries+1)//2 + 1):
      if mu is sigma is None: return
      yield AnnoWarpStitchResultEntry(
        n=n,
        value=mu,
        description=f"constraint mu({description})",
        pscale=self.pscale,
        apscale=self.apscale,
      )
      yield AnnoWarpStitchResultEntry(
        n=n,
        value=sigma,
        description=f"constraint sigma({description})",
        pscale=self.pscale,
        apscale=self.apscale,
      )

  @property
  def allstitchresultentries(self):
    """
    AnnoWarpStitchResultEntries for both the nominal and covariance
    (these are the ones that get written to csv)
    """
    nominal = list(self.stitchresultnominalentries)
    for entry, power in more_itertools.zip_equal(nominal, self.variablepowers()):
      if entry.powerfordescription(entry) != power:
        raise ValueError(f"Wrong power for {entry.description!r}: expected {power}, got {entry.powerfordescription(entry)}")
    return list(itertools.chain(nominal, self.stitchresultcovarianceentries, self.stitchresultconstraintentries))

  @classmethod
  @abc.abstractmethod
  def variablepowers(cls):
    """
    powers of the distance units for the variables (e.g. 1 for pixels, 2 for pixels^2)
    """

  @classmethod
  @abc.abstractmethod
  def nparams(cls):
    """
    number of parameters in the stitching model
    """

  @classmethod
  def floatedparams(cls, floatedparams, mus, sigmas, alignmentresults, logger):
    """
    Returns an array of bools that determine which parameters get floated.
    takes in an array of bools, in which case it returns the input,
    or a string that depends on the model (e.g. "all" for any model,
    or "constants" for the default model)

    Can also depend on alignmentresults to detect degenerate cases.
    """
    if isinstance(floatedparams, str):
      if floatedparams == "all":
        floatedparams = [True] * cls.nparams()
      else:
        raise ValueError(f"Unknown floatedparams {floatedparams!r}")

    if mus is None:
      mus = [None] * cls.nparams()
    if sigmas is None:
      sigmas = [None] * cls.nparams()

    return np.asarray(floatedparams), mus, sigmas

class AnnoWarpStitchResultNoCvxpyBase(AnnoWarpStitchResultBase):
  """
  Stitch result that uses standalone linear algebra and not cvxpy.

  A, b, c: the matrix, vector, and constant that define the quadratic
  to minimize: x^T A x + b^T x + c
  flatresult: the x vector
  """
  def __init__(self, *, A, b, c, flatresult, **kwargs):
    self.A = A
    self.b = b
    self.c = c
    self.flatresult = flatresult
    super().__init__(**kwargs)

  @classmethod
  @abc.abstractmethod
  def unconstrainedAbccontributions(cls, alignmentresult):
    """
    Gives the contributions to A, b, and c from this alignment
    result's residual
    """

  @classmethod
  def constraintAbccontributions(cls, mus, sigmas):
    """
    Gives the contributions to A, b, and c from the constraints.

    mus: means of the gaussian constraints
    sigmas: widths of the gaussian constraints
    """
    if mus is sigmas is None: return 0, 0, 0
    nparams = cls.nparams()
    A = np.zeros(shape=(nparams, nparams), dtype=units.unitdtype)
    b = np.zeros(shape=nparams, dtype=units.unitdtype)
    c = 0
    for i, (mu, sigma) in enumerate(more_itertools.zip_equal(mus, sigmas)):
      #add to negative log likelihood
      #   (x-mu)^2/sigma^2
      # = (1/sigma^2) x^2 - 2 (mu/sigma) x + (mu/sigma)^2
      if mu is sigma is None: continue
      A[i,i] += 1/sigma**2
      b[i] -= 2*mu/sigma**2
      c += (mu/sigma)**2
    return A, b, c

  @classmethod
  def Abc(cls, alignmentresults, mus, sigmas, logger, floatedparams="all", **kwargs):
    """
    Gives the total A, b, and c from the alignment results and constraints.

    alignmentresults: the list of alignment results to use
    mus: means of the gaussian constraints
    sigmas: widths of the gaussian constraints
    floatedparams: which parameters to float
    """
    floatedparams, mus, sigmas = cls.floatedparams(floatedparams, mus, sigmas, alignmentresults, logger=logger)

    mus = np.array(mus)
    sigmas = np.array(sigmas)

    #add the alignment result contributions
    A = b = c = 0
    for alignmentresult in alignmentresults:
      addA, addb, addc = cls.unconstrainedAbccontributions(alignmentresult, **kwargs)
      A += addA
      b += addb
      c += addc

    #if any parameters are fixed, remove the dependence of A and b on those
    #parameters.  The dependence is added to b and c in such a way that, when
    #those parameters are set to the values they're fixed to, the total log
    #likelihood is unchanged.
    floatedindices = np.arange(cls.nparams())[floatedparams]
    fixedindices = np.arange(cls.nparams())[~floatedparams]

    fixedmus = mus[fixedindices]
    fixedsigmas = sigmas[fixedindices]

    badindices = []
    for i, mu, sigma in more_itertools.zip_equal(fixedindices, fixedmus, fixedsigmas):
      if mu is None or sigma is None:
        badindices.append(i)
    if badindices:
      raise ValueError(f"Have to provide non-None constraint mu and sigma for variables #{badindices} if you want to fix them")

    fixedmus = fixedmus.astype(units.unitdtype)
    fixedsigmas = fixedsigmas.astype(units.unitdtype)

    floatfix = np.ix_(floatedindices, fixedindices)
    fixfloat = np.ix_(fixedindices, floatedindices)
    fixfix = np.ix_(fixedindices, fixedindices)

    #A entries that correspond to 2 fixed parameters: goes into c
    c += fixedmus @ A[fixfix] @ fixedmus
    A[fixfix] = 0

    #A entries that correspond to a fixed parameter and a floated parameter
    b[floatedindices] += A[floatfix] @ fixedmus + fixedmus @ A[fixfloat]
    A[floatfix] = A[fixfloat] = 0

    #b entries that correspond to a fixed parameter
    c += b[fixedindices] @ fixedmus
    b[fixedindices] = 0

    #add the constraints
    #the only dependence of A and b on the fixed parameters is the constraints
    #so they end up fitting to mu +/- sigma.  For the floated parameters it's a
    #gaussian constraint but A and b also depend on the alignment results
    addA, addb, addc = cls.constraintAbccontributions(mus, sigmas)

    A += addA
    b += addb
    c += addc

    return A, b, c

class AnnoWarpStitchResultCvxpyBase(AnnoWarpStitchResultBase):
  """
  Stitch result that uses cvxpy.
  This is intended for debugging because it has a nice syntax for setting
  up the minimization in terms of variables.

  problem: the cvxpy Problem object
  """
  def __init__(self, *, problem, **kwargs):
    self.problem = problem
    super().__init__(**kwargs)

  def residual(self, *args, **kwargs):
    """
    The residual for an alignment result.
    We take the nominal value because cvxpy doesn't return the error
    """
    return units.nominal_values(super().residual(*args, **kwargs))

  @classmethod
  @abc.abstractmethod
  def makecvxpyvariables(cls):
    """
    Make the cvxpy Variable objects needed for this class
    """
    return {}

  @classmethod
  @abc.abstractmethod
  def cvxpydxvec(cls, alignmentresult, **cvxpyvariables):
    """
    Get the stitch result dxvec for the alignmentresult as a function of the variables
    """

  @classmethod
  def cvxpyresidual(cls, alignmentresult, **cvxpyvariables):
    """
    Get the residual for the alignmentresult as a function of the variables
    """
    return units.nominal_values(alignmentresult.dxvec)/alignmentresult.oneimpixel - cls.cvxpydxvec(alignmentresult, **cvxpyvariables)

  @classmethod
  def constraintquadforms(cls, cvxpyvariables, mus, sigmas, *, imscale):
    """
    Create the quadratic forms for the constraints
    """
    if mus is sigmas is None: return 0
    onepixel = units.onepixel(imscale)
    result = 0
    musdict = {}
    sigmasdict = {}
    iterator = iter(more_itertools.zip_equal(mus, sigmas, cls.variablepowers(), range(sum(v.size for k, v in cvxpyvariables.items()))))
    #get the mus and sigmas in the right shape
    for name, variable in cvxpyvariables.items():
      musdict[name] = np.zeros(shape=variable.shape)
      sigmasdict[name] = np.zeros(shape=variable.shape)
      raveledmu = musdict[name].ravel()
      raveledsigma = sigmasdict[name].ravel()
      for i, (mu, sigma, power, _) in enumerate(itertools.islice(iterator, int(variable.size))):
        if mu is sigma is None:
          raveledmu[i] = 0
          raveledsigma[i] = float("inf")
        else:
          raveledmu[i] = mu / onepixel**power
          raveledsigma[i] = sigma / onepixel**power

    with np.testing.assert_raises(StopIteration):
      next(iterator)

    #create and sum the quadratic forms ((x-mu)/sigma)^2
    for k, (variable, mu, sigma) in dict_zip_equal(cvxpyvariables, musdict, sigmasdict).items():
      result += cp.sum(((variable-mu)/sigma)**2)

    return result

class AnnoWarpStitchResultDefaultModelBase(AnnoWarpStitchResultBase):
  """
  Stitch result for the default model, which gives \delta\vec{x} as linear
  in the index of the big tile and in the location within the big tile
  """
  def __init__(self, *, coeffrelativetobigtile, bigtileindexcoeff, constant, **kwargs):
    self.coeffrelativetobigtile = coeffrelativetobigtile
    self.bigtileindexcoeff = bigtileindexcoeff
    self.constant = constant
    super().__init__(**kwargs)

  def dxvec(self, qptiffcoordinate, *, apscale):
    """
    Get \Delta\vec{x} for a qptiff coordinate based on the fitted model
    """
    coeffrelativetobigtile = self.coeffrelativetobigtile
    bigtileindexcoeff = units.convertpscale(self.bigtileindexcoeff, self.imscale, apscale)
    constant = units.convertpscale(self.constant, self.imscale, apscale)
    return (
      coeffrelativetobigtile @ qptiffcoordinate.coordinaterelativetobigtile
      + bigtileindexcoeff @ qptiffcoordinate.bigtileindex
      + constant
    )

  @property
  def stitchresultentries(self):
    """
    Get the stitch result entries for the fitted result
    """
    return (
      self.EntryLite(
        value=self.coeffrelativetobigtile[0,0],
        description="coeff dx(x)",
      ),
      self.EntryLite(
        value=self.coeffrelativetobigtile[0,1],
        description="coeff dx(y)",
      ),
      self.EntryLite(
        value=self.coeffrelativetobigtile[1,0],
        description="coeff dy(x)",
      ),
      self.EntryLite(
        value=self.coeffrelativetobigtile[1,1],
        description="coeff dy(y)",
      ),

      self.EntryLite(
        value=self.bigtileindexcoeff[0,0],
        description="coeff dx(ix)",
      ),
      self.EntryLite(
        value=self.bigtileindexcoeff[0,1],
        description="coeff dx(iy)",
      ),
      self.EntryLite(
        value=self.bigtileindexcoeff[1,0],
        description="coeff dy(ix)",
      ),
      self.EntryLite(
        value=self.bigtileindexcoeff[1,1],
        description="coeff dy(iy)",
      ),

      self.EntryLite(
        value=self.constant[0],
        description="const dx",
      ),
      self.EntryLite(
        value=self.constant[1],
        description="const dy",
      ),
    )

  @classmethod
  def variablepowers(cls):
    """
    powers of the distance units for the variables
    coeffrelativetobigtile is dimensionless, bigtileindexcoeff and constant
    have units of distance
    """
    return 0, 0, 0, 0, 1, 1, 1, 1, 1, 1

  @classmethod
  def nparams(cls):
    return 10

  @classmethod
  def floatedparams(cls, floatedparams, mus, sigmas, alignmentresults, logger):
    if isinstance(floatedparams, str):
      if floatedparams == "constants":
        floatedparams = [False]*8+[True]*2
        if mus is sigmas is None:
          mus = [0] * 8 + [None] * 2
          sigmas = [1e-10 * alignmentresults[0].onepixel] * 8 + [None] * 2

    floatedparams, mus, sigmas = super().floatedparams(floatedparams, mus, sigmas, alignmentresults, logger=logger)

    bigtileindices = np.array([_.bigtileindex for _ in alignmentresults])
    bigtilexs, bigtileys = bigtileindices.T
    if len(set(bigtilexs)) == 1:
      floatedparams[4] = floatedparams[6] = False
      if mus[4] is None: mus[4] = 0
      if sigmas[4] is None: sigmas[4] = .001*alignmentresults[0].oneimpixel
      if mus[6] is None: mus[6] = 0
      if sigmas[6] is None: sigmas[6] = .001*alignmentresults[0].oneimpixel
    if len(set(bigtileys)) == 1:
      floatedparams[5] = floatedparams[7] = False
      if mus[5] is None: mus[5] = 0
      if sigmas[5] is None: sigmas[5] = .001*alignmentresults[0].oneimpixel
      if mus[7] is None: mus[7] = 0
      if sigmas[7] is None: sigmas[7] = .001*alignmentresults[0].oneimpixel

    return floatedparams, mus, sigmas

class AnnoWarpStitchResultDefaultModelWithJumpsBase(AnnoWarpStitchResultDefaultModelBase):
  @classmethod
  @abc.abstractmethod
  def xdxjumppositions(cls): pass
  @classmethod
  def nxdxjumps(cls): return len(cls.xdxjumppositions())
  @classmethod
  @abc.abstractmethod
  def xdyjumppositions(cls): pass
  @classmethod
  def nxdyjumps(cls): return len(cls.xdyjumppositions())
  @classmethod
  @abc.abstractmethod
  def ydxjumppositions(cls): pass
  @classmethod
  def nydxjumps(cls): return len(cls.ydxjumppositions())
  @classmethod
  @abc.abstractmethod
  def ydyjumppositions(cls): pass
  @classmethod
  def nydyjumps(cls): return len(cls.ydyjumppositions())
  @classmethod
  def ntotaljumps(cls):
    return cls.nxdxjumps() + cls.nxdyjumps() + cls.nydxjumps() + cls.nydyjumps()

  @classmethod
  def subclass(cls, *, xdxjumppositions, xdyjumppositions, ydxjumppositions, ydyjumppositions):
    class subcls(cls):
      @classmethod
      def xdxjumppositions(cls): return xdxjumppositions
      @classmethod
      def xdyjumppositions(cls): return xdyjumppositions
      @classmethod
      def ydxjumppositions(cls): return ydxjumppositions
      @classmethod
      def ydyjumppositions(cls): return ydyjumppositions
    name = cls.__name__
    name += "_xdx_" + "_".join(str(xdxjumpposition) for xdxjumpposition in xdxjumppositions)
    name += "_xdy_" + "_".join(str(xdyjumpposition) for xdyjumpposition in xdyjumppositions)
    name += "_ydx_" + "_".join(str(ydxjumpposition) for ydxjumpposition in ydxjumppositions)
    name += "_ydy_" + "_".join(str(ydyjumpposition) for ydyjumpposition in ydyjumppositions)
    name = name.replace(".", "p")
    name = name.replace("-", "m")
    name = name.replace("+", "p")
    match = re.match(r"^\w+$", name, flags=re.ASCII)
    if not match: raise ValueError(f"{name!r} doesn't match {match!r}")
    subcls.__name__ = name
    return subcls

  def __init__(self, *, xdxjumps, xdyjumps, ydxjumps, ydyjumps, **kwargs):
    self.xdxjumps = xdxjumps
    self.xdyjumps = xdyjumps
    self.ydxjumps = ydxjumps
    self.ydyjumps = ydyjumps

    if len(self.xdxjumps) != self.nxdxjumps():
      raise ValueError(f"Mismatch in xdxjumps: {len(self.xdxjumps)} {self.nxdxjumps}")
    if len(self.xdyjumps) != self.nxdyjumps():
      raise ValueError(f"Mismatch in xdyjumps: {len(self.xdyjumps)} {self.nxdyjumps}")
    if len(self.ydxjumps) != self.nydxjumps():
      raise ValueError(f"Mismatch in ydxjumps: {len(self.ydxjumps)} {self.nydxjumps}")
    if len(self.ydyjumps) != self.nydyjumps():
      raise ValueError(f"Mismatch in ydyjumps: {len(self.ydyjumps)} {self.nydyjumps}")

    super().__init__(**kwargs)

  def dxvec(self, qptiffcoordinate, *, apscale):
    """
    Get \Delta\vec{x} for a qptiff coordinate based on the fitted model
    """
    x, y = qptiffcoordinate.xvec
    dx, dy = super().dxvec(qptiffcoordinate, apscale=apscale)

    xdxjumps = units.convertpscale(self.xdxjumps, self.imscale, apscale)
    xdxjumppositions = units.convertpscale(self.xdxjumppositions(), self.imscale, apscale)
    xdyjumps = units.convertpscale(self.xdyjumps, self.imscale, apscale)
    xdyjumppositions = units.convertpscale(self.xdyjumppositions(), self.imscale, apscale)
    ydxjumps = units.convertpscale(self.ydxjumps, self.imscale, apscale)
    ydxjumppositions = units.convertpscale(self.ydxjumppositions(), self.imscale, apscale)
    ydyjumps = units.convertpscale(self.ydyjumps, self.imscale, apscale)
    ydyjumppositions = units.convertpscale(self.ydyjumppositions(), self.imscale, apscale)
    for xdxjumpposition, xdxjump in more_itertools.zip_equal(xdxjumppositions, xdxjumps):
      if x > xdxjumpposition: dx += xdxjump
    for xdyjumpposition, xdyjump in more_itertools.zip_equal(xdyjumppositions, xdyjumps):
      if x > xdyjumpposition: dy += xdyjump
    for ydxjumpposition, ydxjump in more_itertools.zip_equal(ydxjumppositions, ydxjumps):
      if y > ydxjumpposition: dx += ydxjump
    for ydyjumpposition, ydyjump in more_itertools.zip_equal(ydyjumppositions, ydyjumps):
      if y > ydyjumpposition: dy += ydyjump

    return np.array([dx, dy])

  @property
  def stitchresultentries(self):
    """
    Get the stitch result entries for the fitted result
    """
    return (
      *super().stitchresultentries,
      *sum((
        (
          self.EntryLite(
            value=xdxjumpposition,
            description=f"position of dx jump #{i} in x",
          ), 
          self.EntryLite(
            value=xdxjump,
            description=f"dx jump #{i} in x",
          ),
        ) for i, (xdxjumpposition, xdxjump) in enumerate(more_itertools.zip_equal(self.xdxjumppositions(), self.xdxjumps))
      ), ()),
      *sum((
        (
          self.EntryLite(
            value=xdyjumpposition,
            description=f"position of dy jump #{i} in x",
          ), 
          self.EntryLite(
            value=xdyjump,
            description=f"dy jump #{i} in x",
          ),
        ) for i, (xdyjumpposition, xdyjump) in enumerate(more_itertools.zip_equal(self.xdyjumppositions(), self.xdyjumps))
      ), ()),
      *sum((
        (
          self.EntryLite(
            value=ydxjumpposition,
            description=f"position of dx jump #{i} in y",
          ), 
          self.EntryLite(
            value=ydxjump,
            description=f"dx jump #{i} in y",
          ),
        ) for i, (ydxjumpposition, ydxjump) in enumerate(more_itertools.zip_equal(self.ydxjumppositions(), self.ydxjumps))
      ), ()),
      *sum((
        (
          self.EntryLite(
            value=ydyjumpposition,
            description=f"position of dy jump #{i} in y",
          ), 
          self.EntryLite(
            value=ydyjump,
            description=f"dy jump #{i} in y",
          ),
        ) for i, (ydyjumpposition, ydyjump) in enumerate(more_itertools.zip_equal(self.ydyjumppositions(), self.ydyjumps))
      ), ()),
    )

  @classmethod
  def variablepowers(cls, **kwargs):
    """
    powers of the distance units for the variables
    coeffrelativetobigtile is dimensionless, bigtileindexcoeff and constant
    have units of distance
    """
    return super().variablepowers(**kwargs) + (1,) * cls.ntotaljumps()

  @classmethod
  def nparams(cls, **kwargs):
    return super().nparams(**kwargs) + cls.ntotaljumps()

  @classmethod
  def floatedparams(cls, floatedparams, mus, sigmas, alignmentresults, logger, **kwargs):
    floatedparams, mus, sigmas = super().floatedparams(floatedparams, mus, sigmas, alignmentresults, logger=logger)

    if len(floatedparams) == cls.nparams():
      pass
    elif len(floatedparams) == super().nparams():
      floatedparams = list(floatedparams) + [True] * cls.ntotaljumps()

    if len(floatedparams) != cls.nparams():
      raise ValueError(f"floatedparams has the wrong length {len(floatedparams)}")

    return floatedparams, mus, sigmas

class AnnoWarpStitchResultDefaultModel(AnnoWarpStitchResultDefaultModelBase, AnnoWarpStitchResultNoCvxpyBase):
  """
  Stitch result for the default model with no cvxpy
  """
  def __init__(self, flatresult, **kwargs):
    coeffrelativetobigtile, bigtileindexcoeff, constant, other = np.split(flatresult, [4, 8, 10])
    coeffrelativetobigtile = coeffrelativetobigtile.reshape(2, 2)
    bigtileindexcoeff = bigtileindexcoeff.reshape(2, 2)
    super().__init__(flatresult=flatresult, coeffrelativetobigtile=coeffrelativetobigtile, bigtileindexcoeff=bigtileindexcoeff, constant=constant, **kwargs)

  @classmethod
  def unconstrainedAbccontributions(cls, alignmentresult):
    """
    Assemble the A matrix, b vector, and c scalar for the default model
    from the alignment result
    """

    nparams = cls.nparams()
    #get the indices for each parameter
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

    #create A, b, and c
    A = np.zeros(shape=(nparams, nparams), dtype=units.unitdtype)
    b = np.zeros(shape=nparams, dtype=units.unitdtype)
    c = 0

    #get the factors that the parameters are going to multiply
    crtbt = alignmentresult.coordinaterelativetobigtile
    bti = alignmentresult.bigtileindex

    #get the alignment result dxvec and covariance matrix
    dxvec = units.nominal_values(alignmentresult.dxvec)
    invcov = units.np.linalg.inv(alignmentresult.covariance)

    #fill the A matrix
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

    #fill the b vector
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

    #fill c
    c += dxvec @ invcov @ dxvec

    return A, b, c

class AnnoWarpStitchResultDefaultModelCvxpy(AnnoWarpStitchResultDefaultModelBase, AnnoWarpStitchResultCvxpyBase):
  """
  Stitch result for the default model with cvxpy
  """
  def __init__(self, *, coeffrelativetobigtile, bigtileindexcoeff, constant, pscale, apscale, **kwargs):
    onepixel = units.onepixel(pscale=pscale / np.round(float(pscale/apscale)))
    super().__init__(
      coeffrelativetobigtile=coeffrelativetobigtile.value,
      bigtileindexcoeff=bigtileindexcoeff.value * onepixel,
      constant=constant.value * onepixel,
      pscale=pscale,
      apscale=apscale,
      **kwargs,
    )
    self.coeffrelativetobigtilevar = coeffrelativetobigtile
    self.bigtileindexcoeffvar = bigtileindexcoeff
    self.constantvar = constant

  @classmethod
  def makecvxpyvariables(cls):
    return {
      "coeffrelativetobigtile": cp.Variable(shape=(2, 2)),
      "bigtileindexcoeff": cp.Variable(shape=(2, 2)),
      "constant": cp.Variable(shape=2),
    }

  @classmethod
  def cvxpydxvec(cls, alignmentresult, *, coeffrelativetobigtile, bigtileindexcoeff, constant):
    return (
      coeffrelativetobigtile @ (alignmentresult.coordinaterelativetobigtile / alignmentresult.oneimpixel)
      + bigtileindexcoeff @ alignmentresult.bigtileindex
      + constant
    )

class AnnoWarpStitchResultDefaultModelWithJumps(AnnoWarpStitchResultDefaultModelWithJumpsBase, AnnoWarpStitchResultDefaultModel, AnnoWarpStitchResultNoCvxpyBase):
  def __init__(self, flatresult, **kwargs):
    other, xdxjumps, xdyjumps, ydxjumps, ydyjumps = np.split(
      flatresult, [
        10,
        10+self.nxdxjumps(),
        10+self.nxdxjumps()+self.nxdyjumps(),
        10+self.nxdxjumps()+self.nxdyjumps()+self.nydxjumps(),
      ]
    )
    super().__init__(flatresult=flatresult, xdxjumps=xdxjumps, xdyjumps=xdyjumps, ydxjumps=ydxjumps, ydyjumps=ydyjumps, **kwargs)

  @classmethod
  def unconstrainedAbccontributions(cls, alignmentresult, *, _debug=True):
    """
    Assemble the A matrix, b vector, and c scalar for the default model
    from the alignment result
    """
    nparams = cls.nparams()
    #get the indices for each parameter
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
      *jumps,
    ) = range(nparams)

    xdxjumps, xdyjumps, ydxjumps, ydyjumps = np.split(jumps, [cls.nxdxjumps(), cls.nxdxjumps()+cls.nxdyjumps(), cls.nxdxjumps()+cls.nxdyjumps()+cls.nydxjumps()])

    const_x_indices = [const_x]
    const_y_indices = [const_y]

    for xdxjumpposition, xdxjump in more_itertools.zip_equal(cls.xdxjumppositions(), xdxjumps):
      if alignmentresult.x > xdxjumpposition:
        const_x_indices.append(xdxjump)
    for xdyjumpposition, xdyjump in more_itertools.zip_equal(cls.xdyjumppositions(), xdyjumps):
      if alignmentresult.x > xdyjumpposition:
        const_y_indices.append(xdyjump)
    for ydxjumpposition, ydxjump in more_itertools.zip_equal(cls.ydxjumppositions(), ydxjumps):
      if alignmentresult.y > ydxjumpposition:
        const_x_indices.append(ydxjump)
    for ydyjumpposition, ydyjump in more_itertools.zip_equal(cls.ydyjumppositions(), ydyjumps):
      if alignmentresult.y > ydyjumpposition:
        const_y_indices.append(ydyjump)

    #create A, b, and c
    A = np.zeros(shape=(nparams, nparams), dtype=units.unitdtype)
    b = np.zeros(shape=nparams, dtype=units.unitdtype)
    c = 0

    #get the factors that the parameters are going to multiply
    crtbt = alignmentresult.coordinaterelativetobigtile
    bti = alignmentresult.bigtileindex

    #get the alignment result dxvec and covariance matrix
    dxvec = units.nominal_values(alignmentresult.dxvec)
    invcov = units.np.linalg.inv(alignmentresult.covariance)

    #fill the A matrix
    A[crtbt_xx:crtbt_xy+1, crtbt_xx:crtbt_xy+1] += np.outer(crtbt, crtbt) * invcov[0,0]
    A[crtbt_yx:crtbt_yy+1, crtbt_xx:crtbt_xy+1] += np.outer(crtbt, crtbt) * invcov[0,1]
    A[crtbt_xx:crtbt_xy+1, crtbt_yx:crtbt_yy+1] += np.outer(crtbt, crtbt) * invcov[1,0]
    A[crtbt_yx:crtbt_yy+1, crtbt_yx:crtbt_yy+1] += np.outer(crtbt, crtbt) * invcov[1,1]

    A[crtbt_xx:crtbt_xy+1, bti_xx:bti_xy+1] += np.outer(crtbt, bti) * invcov[0,0]
    A[crtbt_yx:crtbt_yy+1, bti_xx:bti_xy+1] += np.outer(crtbt, bti) * invcov[0,1]
    A[crtbt_xx:crtbt_xy+1, bti_yx:bti_yy+1] += np.outer(crtbt, bti) * invcov[1,0]
    A[crtbt_yx:crtbt_yy+1, bti_yx:bti_yy+1] += np.outer(crtbt, bti) * invcov[1,1]

    for const_x_idx in const_x_indices:
      A[crtbt_xx:crtbt_xy+1, const_x_idx] += crtbt * invcov[0,0]
      A[crtbt_yx:crtbt_yy+1, const_x_idx] += crtbt * invcov[1,0]
    for const_y_idx in const_y_indices:
      A[crtbt_xx:crtbt_xy+1, const_y_idx] += crtbt * invcov[0,1]
      A[crtbt_yx:crtbt_yy+1, const_y_idx] += crtbt * invcov[1,1]

    A[bti_xx:bti_xy+1, crtbt_xx:crtbt_xy+1] += np.outer(bti, crtbt) * invcov[0,0]
    A[bti_yx:bti_yy+1, crtbt_xx:crtbt_xy+1] += np.outer(bti, crtbt) * invcov[0,1]
    A[bti_xx:bti_xy+1, crtbt_yx:crtbt_yy+1] += np.outer(bti, crtbt) * invcov[1,0]
    A[bti_yx:bti_yy+1, crtbt_yx:crtbt_yy+1] += np.outer(bti, crtbt) * invcov[1,1]

    A[bti_xx:bti_xy+1, bti_xx:bti_xy+1] += np.outer(bti, bti) * invcov[0,0]
    A[bti_yx:bti_yy+1, bti_xx:bti_xy+1] += np.outer(bti, bti) * invcov[0,1]
    A[bti_xx:bti_xy+1, bti_yx:bti_yy+1] += np.outer(bti, bti) * invcov[1,0]
    A[bti_yx:bti_yy+1, bti_yx:bti_yy+1] += np.outer(bti, bti) * invcov[1,1]

    for const_x_idx in const_x_indices:
      A[bti_xx:bti_xy+1, const_x_idx] += bti * invcov[0,0]
      A[bti_yx:bti_yy+1, const_x_idx] += bti * invcov[1,0]
    for const_y_idx in const_y_indices:
      A[bti_xx:bti_xy+1, const_y_idx] += bti * invcov[0,1]
      A[bti_yx:bti_yy+1, const_y_idx] += bti * invcov[1,1]

    for const_x_idx in const_x_indices:
      A[const_x_idx, crtbt_xx:crtbt_xy+1] += crtbt * invcov[0,0]
      A[const_x_idx, crtbt_yx:crtbt_yy+1] += crtbt * invcov[1,0]
    for const_y_idx in const_y_indices:
      A[const_y_idx, crtbt_xx:crtbt_xy+1] += crtbt * invcov[0,1]
      A[const_y_idx, crtbt_yx:crtbt_yy+1] += crtbt * invcov[1,1]

    for const_x_idx in const_x_indices:
      A[const_x_idx, bti_xx:bti_xy+1] += bti * invcov[0,0]
      A[const_x_idx, bti_yx:bti_yy+1] += bti * invcov[1,0]
    for const_y_idx in const_y_indices:
      A[const_y_idx, bti_xx:bti_xy+1] += bti * invcov[0,1]
      A[const_y_idx, bti_yx:bti_yy+1] += bti * invcov[1,1]

    for const_x_idx_1 in const_x_indices:
      for const_x_idx_2 in const_x_indices:
        A[const_x_idx_1, const_x_idx_2] += invcov[0,0]
    for const_x_idx in const_x_indices:
      for const_y_idx in const_y_indices:
        A[const_x_idx, const_y_idx] += invcov[0,1]
        A[const_y_idx, const_x_idx] += invcov[1,0]
    for const_y_idx_1 in const_y_indices:
      for const_y_idx_2 in const_y_indices:
        A[const_y_idx_1, const_y_idx_2] += invcov[1,1]

    #fill the b vector
    b[crtbt_xx:crtbt_xy+1] -= 2 * crtbt * invcov[0,0] * dxvec[0]
    b[crtbt_xx:crtbt_xy+1] -= 2 * crtbt * invcov[0,1] * dxvec[1]
    b[crtbt_yx:crtbt_yy+1] -= 2 * crtbt * invcov[1,0] * dxvec[0]
    b[crtbt_yx:crtbt_yy+1] -= 2 * crtbt * invcov[1,1] * dxvec[1]

    b[bti_xx:bti_xy+1] -= 2 * bti * invcov[0,0] * dxvec[0]
    b[bti_xx:bti_xy+1] -= 2 * bti * invcov[0,1] * dxvec[1]
    b[bti_yx:bti_yy+1] -= 2 * bti * invcov[1,0] * dxvec[0]
    b[bti_yx:bti_yy+1] -= 2 * bti * invcov[1,1] * dxvec[1]

    for const_x_idx in const_x_indices:
      b[const_x_idx] -= 2 * invcov[0,0] * dxvec[0]
      b[const_x_idx] -= 2 * invcov[0,1] * dxvec[1]
    for const_y_idx in const_y_indices:
      b[const_y_idx] -= 2 * invcov[1,0] * dxvec[0]
      b[const_y_idx] -= 2 * invcov[1,1] * dxvec[1]

    #fill c
    c += dxvec @ invcov @ dxvec

    if _debug:
      superA, superb, superc = AnnoWarpStitchResultDefaultModel.unconstrainedAbccontributions(alignmentresult) #not super because that will give the wrong nparams
      np.testing.assert_array_equal(A[:10,:10], superA)
      np.testing.assert_array_equal(b[:10], superb)
      np.testing.assert_array_equal(c, superc)

    return A, b, c

class AnnoWarpStitchResultDefaultModelWithJumpsCvxpy(AnnoWarpStitchResultDefaultModelWithJumpsBase, AnnoWarpStitchResultDefaultModelCvxpy, AnnoWarpStitchResultCvxpyBase):
  def __init__(self, *, xdxjumps, xdyjumps, ydxjumps, ydyjumps, pscale, apscale, **kwargs):
    onepixel = units.onepixel(pscale=pscale / np.round(float(pscale/apscale)))
    super().__init__(
      xdxjumps=xdxjumps.value*onepixel,
      xdyjumps=xdyjumps.value*onepixel,
      ydxjumps=ydxjumps.value*onepixel,
      ydyjumps=ydyjumps.value*onepixel,
      pscale=pscale,
      apscale=apscale,
      **kwargs,
    )
    self.xdxjumpsvar = xdxjumps
    self.xdyjumpsvar = xdyjumps
    self.ydxjumpsvar = ydxjumps
    self.ydyjumpsvar = ydyjumps

  @classmethod
  def makecvxpyvariables(cls):
    return {
      **super().makecvxpyvariables(),
      "xdxjumps": cp.Variable(shape=cls.nxdxjumps()),
      "xdyjumps": cp.Variable(shape=cls.nxdyjumps()),
      "ydxjumps": cp.Variable(shape=cls.nydxjumps()),
      "ydyjumps": cp.Variable(shape=cls.nydyjumps()),
    }

  @classmethod
  def cvxpydxvec(cls, alignmentresult, *, xdxjumps, xdyjumps, ydxjumps, ydyjumps, **kwargs):
    result = super().cvxpydxvec(alignmentresult, **kwargs)
    dx = dy = 0
    for xdxjumpposition, xdxjump in more_itertools.zip_equal(cls.xdxjumppositions(), xdxjumps):
      if alignmentresult.x > xdxjumpposition:
        dx += xdxjump
    for xdyjumpposition, xdyjump in more_itertools.zip_equal(cls.xdyjumppositions(), xdyjumps):
      if alignmentresult.x > xdyjumpposition:
        dy += xdyjump
    for ydxjumpposition, ydxjump in more_itertools.zip_equal(cls.ydxjumppositions(), ydxjumps):
      if alignmentresult.y > ydxjumpposition:
        dx += ydxjump
    for ydyjumpposition, ydyjump in more_itertools.zip_equal(cls.ydyjumppositions(), ydyjumps):
      if alignmentresult.y > ydyjumpposition:
        dy += ydyjump
    result += np.array([dx, dy])
    return result

class AnnoWarpStitchResultEntry(DataClassWithImscale):
  """
  Stitch result entry dataclass for the csv file

  n: numerical index of the entry
  value: the value of the parameter or covariance entry
  description: description in words
  """
  @classmethod
  def powerfordescription(cls, self_or_description):
    if isinstance(self_or_description, cls):
      description = self_or_description.description
    else:
      description = self_or_description
    dct = {
      "coeff dx(x)": 0,
      "coeff dx(y)": 0,
      "coeff dy(x)": 0,
      "coeff dy(y)": 0,
      "coeff dx(ix)": 1,
      "coeff dx(iy)": 1,
      "coeff dy(ix)": 1,
      "coeff dy(iy)": 1,
      "const dx": 1,
      "const dy": 1,
    }
    covmatch = re.match(r"cov\((.*), (.*)\)$", description)
    constraint_match = re.match(r"constraint (?:mu|sigma)\((.*)\)$", description)
    if covmatch:
      return dct[covmatch.group(1)] + dct[covmatch.group(2)]
    elif constraint_match:
      return dct[constraint_match.group(1)]
    else:
      return dct[description]
  n: int
  value: units.Distance = distancefield(pixelsormicrons="pixels", power=lambda self: self.powerfordescription(self), pscalename="imscale")
  description: str
