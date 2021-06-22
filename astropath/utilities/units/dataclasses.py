import dataclassy, functools, methodtools, numbers, numpy as np
from ..dataclasses import MetaDataAnnotation, MyDataClass, MyDataClassFrozen
from ..misc import floattoint
from .core import ThingWithApscale, ThingWithImscale, ThingWithPscale, ThingWithQpscale, UnitsError

def __setup(mode):
  global currentmode, microns, pixels, _pscale, safe, UnitsError
  from . import safe as safe
  if mode == "safe":
    from .safe import microns, pixels
    from .safe.core import _pscale
  elif mode == "fast":
    from .fast import microns, pixels
    def _pscale(distance): return None
  else:
    raise ValueError(f"Invalid mode {mode}")
  currentmode = mode

__notgiven = object()
def distancefield(*defaultvalue, pixelsormicrons, power=1, dtype=float, secondfunction=None, pscalename="pscale", **metadata):
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
  return MetaDataAnnotation(*defaultvalue, **metadata)

def pscalefield(*defaultvalue, **metadata):
  metadata = {
    "includeintable": False,
    "ispscalefield": True,
    **metadata,
  }
  return MetaDataAnnotation(*defaultvalue, **metadata)

class DataClassWithDistances(MyDataClass):
  @methodtools.lru_cache()
  @classmethod
  def distancefields(cls):
    return [field for field in dataclassy.fields(cls) if cls.metadata(field).get("isdistancefield", False)]

  @methodtools.lru_cache()
  @classmethod
  def pscalefields(cls):
    return [field for field in dataclassy.fields(cls) if cls.metadata(field).get("ispscalefield", False)]

  def _distances_passed_to_init(self):
    return [getattr(self, fieldname) for fieldname in self.distancefields()]

  def __post_init__(self, *args, readingfromfile=False, **kwargs):
    powers = {}
    pscalenames = {}
    types = dataclassy.fields(self)

    for field in self.distancefields():
      value = getattr(self, field)
      if isinstance(value, np.ndarray) and not value.shape:
        setattr(self, field, value[()])

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
      if distances and any(distances):
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

    super().__post_init__(*args, readingfromfile=readingfromfile, **kwargs)

def makedataclasswithpscale(classname, pscalename, thingwithpscalecls):
  class cls(DataClassWithDistances, thingwithpscalecls): pass
  cls.__name__ = classname
  cls.__annotations__[pscalename] = float
  varname = f"_{classname}__{pscalename}"
  def getter(self): return getattr(self, varname)
  def setter(self, pscale): setattr(self, varname, pscale)
  setattr(cls, pscalename, pscalefield(property(getter, setter)))
  cls = dataclassy.dataclass(cls)
  cls.__defaults__.pop(pscalename)

  class frozen(cls, MyDataClassFrozen): pass
  frozen.__name__ = f"{classname}Frozen"

  return cls, frozen

DataClassWithPscale, DataClassWithPscaleFrozen = makedataclasswithpscale("DataClassWithPscale", "pscale", ThingWithPscale)
DataClassWithQpscale, DataClassWithQpscaleFrozen = makedataclasswithpscale("DataClassWithQpscale", "qpscale", ThingWithQpscale)
DataClassWithApscale, DataClassWithApscaleFrozen = makedataclasswithpscale("DataClassWithApscale", "apscale", ThingWithApscale)
DataClassWithImscale, DataClassWithImscaleFrozen = makedataclasswithpscale("DataClassWithImscale", "imscale", ThingWithImscale)
