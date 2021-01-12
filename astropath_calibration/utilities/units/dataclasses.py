import abc, dataclassy, functools, numbers, numpy as np
from ..dataclasses import MetaDataAnnotation, MyDataClass
from ..misc import floattoint
from .core import Distance, ThingWithApscale, ThingWithPscale, ThingWithQpscale, UnitsError

def __setup(mode):
  global currentmode, Distance, microns, pixels, _pscale, safe, UnitsError
  from . import safe as safe
  if mode == "safe":
    from .safe import Distance, microns, pixels
    from .safe.core import _pscale
  elif mode == "fast":
    from .fast import Distance, microns, pixels
    def _pscale(distance): return None
  else:
    raise ValueError(f"Invalid mode {mode}")
  currentmode = mode

def distancefield(*, pixelsormicrons, typ=Distance, power=1, dtype=float, secondfunction=None, pscalename="pscale", **metadata):
  if secondfunction is None:
    if issubclass(dtype, numbers.Integral):
      secondfunction = functools.partial(floattoint, atol=1e-9)
    else:
      secondfunction = lambda x: x

  if not callable(pixelsormicrons):
    _pixelsormicrons = pixelsormicrons
    pixelsormicrons = lambda *args, **kwargs: _pixelsormicrons

  if not callable(power):
    _power = power
    power = lambda *args, **kwargs: _power

  if not callable(pscalename):
    _pscalename = pscalename
    pscalename = lambda *args, **kwargs: _pscalename

  metadata = {
    "writefunction": lambda *args, pixelsormicrons, **kwargs: secondfunction({
      "pixels": pixels,
      "microns": microns,
    }[pixelsormicrons](*args, **kwargs)),
    "readfunction": dtype,
    "isdistancefield": True,
    "pixelsormicrons": pixelsormicrons,
    "power": power,
    "pscalename": pscalename,
    "writefunctionkwargs": lambda object: {"pscale": getattr(object, pscalename(object)), "power": power(object), "pixelsormicrons": pixelsormicrons(object)},
    **metadata,
  }
  return MetaDataAnnotation(typ, **metadata)

def pscalefield(typ=float, **metadata):
  metadata = {
    "includeintable": False,
    "ispscalefield": True,
    **metadata,
  }
  return MetaDataAnnotation(typ, **metadata)

class DataClassWithDistances(MyDataClass):
  @classmethod
  def distancefields(cls):
    return [field for field in dataclassy.fields(cls) if cls.metadata(field).get("isdistancefield", False)]

  @classmethod
  def pscalefields(cls):
    return [field for field in dataclassy.fields(cls) if cls.metadata(field).get("ispscalefield", False)]

  def _distances_passed_to_init(self):
    return [getattr(self, fieldname) for fieldname in self.distancefields()]

  def __init__(self, readingfromfile=False):
    powers = {}
    pscalenames = {}
    types = dataclassy.fields(self)

    for field in self.distancefields():
      value = getattr(self, field)
      if isinstance(value, np.ndarray) and not value.shape:
        setattr(self, field.name, value[()])

    for fieldname in self.distancefields():
      power = self.metadata(fieldname)["power"]
      if callable(power):
        power = power(self)
      if not isinstance(power, numbers.Number):
        raise TypeError(f"power should be a number or a function, not {type(power)}")
      powers[fieldname] = power

      pscalename = self.metadata(fieldname)["pscalename"]
      if callable(pscalename):
        pscalename = pscalename(self)
      if not isinstance(pscalename, str):
        raise TypeError(f"pscalename should be a number or a function, not {type(pscalename)}")
      pscalenames[fieldname] = pscalename

    usedistances = False
    if currentmode == "safe" and any(powers.values()):
      distances = self._distances_passed_to_init()
      if distances:
        try:
          usedistances, = {isinstance(_, safe.Distance) for _ in distances if _}
        except ValueError:
          raise ValueError(f"Provided some distances and some pixels/microns to {type(self).__name__} - this is dangerous!\n{distances}")
        if usedistances and readingfromfile: assert False #shouldn't be able to happen
        if not usedistances and not readingfromfile:
          raise ValueError("Have to init with readingfromfile=True if you're not providing distances")
      else:
        usedistances = False

    pscales = {}
    for pscalefieldname in self.pscalefields():
      pscale = {getattr(self, pscalefieldname)}
      distancefieldnames = [distancefieldname for distancefieldname in self.distancefields() if pscalenames[distancefieldname] == pscalefieldname]
      nonzerodistancefieldnames = [distancefieldname for distancefieldname in distancefieldnames if getattr(self, distancefieldname)]
      if usedistances and nonzerodistancefieldnames:
        pscale |= set(_pscale([getattr(self, distancefieldname) for distancefieldname in nonzerodistancefieldnames]))
      pscale.discard(None)
      if len(pscale) == 1:
        pscale, = pscale
      elif not any(powers[distancefieldname] for distancefieldname in nonzerodistancefieldnames):
        pscale = None
      elif not pscale:
        raise TypeError(f"Have to either provide {pscalefieldname} explicitly or give coordinates in units.Distance form")
      elif len(pscale) > 1:
        raise UnitsError(f"Provided inconsistent pscales {pscale} for {pscalefieldname}")
      else:
        assert False, "This can't happen"

      for distancefieldname in distancefieldnames:
        pscales[distancefieldname] = pscale

    if readingfromfile:
      for fieldname in self.distancefields():
        setattr(self, fieldname, types[fieldname](power=powers[fieldname], pscale=pscales[fieldname], **{self.metadata(fieldname)["pixelsormicrons"](self): getattr(self, fieldname)}))

class DataClassWithPscale(DataClassWithDistances, ThingWithPscale):
  pscale: pscalefield(float)

class DataClassWithQpscale(DataClassWithDistances, ThingWithQpscale):
  qpscale: pscalefield(float)

class DataClassWithApscale(DataClassWithDistances, ThingWithApscale):
  apscale: pscalefield(float)
