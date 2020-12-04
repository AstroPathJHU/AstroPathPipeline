import dataclasses, functools, numbers
from ..misc import floattoint
from .core import ThingWithPscale, UnitsError

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

def distancefield(pixelsormicrons, *, metadata={}, power=1, dtype=float, secondfunction=None, pscalename="pscale", **kwargs):
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

  kwargs["metadata"] = {
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
  return dataclasses.field(**kwargs)

def pscalefield(*, metadata={}, **kwargs):
  kwargs["metadata"] = {
    "includeintable": False,
    "ispscalefield": True,
    **metadata,
  }
  return dataclasses.field(**kwargs)

@dataclasses.dataclass
class DataClassWithDistances(ThingWithPscale):
  @classmethod
  def distancefields(cls):
    return [field for field in dataclasses.fields(cls) if field.metadata.get("isdistancefield", False)]

  @classmethod
  def pscalefields(cls):
    return [field for field in dataclasses.fields(cls) if field.metadata.get("ispscalefield", False)]

  def _distances_passed_to_init(self):
    return [getattr(self, _.name) for _ in self.distancefields()]

  def __post_init__(self, readingfromfile=False):
    powers = {}
    pscalenames = {}
    for field in self.distancefields():
      power = field.metadata["power"]
      if callable(power):
        power = power(self)
      if not isinstance(power, numbers.Number):
        raise TypeError(f"power should be a number or a function, not {type(power)}")
      powers[field.name] = power

      pscalename = field.metadata["pscalename"]
      if callable(pscalename):
        pscalename = pscalename(self)
      if not isinstance(pscalename, str):
        raise TypeError(f"pscalename should be a number or a function, not {type(pscalename)}")
      pscalenames[field.name] = pscalename

    usedistances = False
    if currentmode == "safe" and any(powers.values()):
      distances = self._distances_passed_to_init()
      if distances:
        try:
          usedistances = {isinstance(_, safe.Distance) for _ in distances if _}
        except ValueError:
          raise ValueError(f"Provided some distances and some pixels/microns to {type(self).__name__} - this is dangerous!")
        if usedistances and readingfromfile: assert False #shouldn't be able to happen
        if not usedistances and not readingfromfile:
          raise ValueError("Have to init with readingfromfile=True if you're not providing distances")
      else:
        usedistances = False

    pscales = {}
    for pscalefield in self.pscalefields():
      pscale = {getattr(self, pscalefield.name)}
      distancefields = [distancefield for distancefield in self.distancefields() if pscalenames[distancefield.name] == pscalefield.name and getattr(self, distancefield.name)]
      if usedistances:
        pscale |= set(_pscale([getattr(self, distancefield.name) for distancefield in distancefields]))
      pscale.discard(None)
      if len(pscale) == 1:
        pscale, = pscale
      elif not any(powers[distancefield.name] for distancefield in distancefields):
        pscale = None
      elif not pscale:
        raise TypeError(f"Have to either provide {pscalefield.name} explicitly or give coordinates in units.Distance form")
      elif len(pscale) > 1:
        raise UnitsError(f"Provided inconsistent pscales {pscale} for {pscalefield.name}")
      else:
        assert False, "This can't happen"

      for distancefield in distancefields:
        pscales[distancefield.name] = pscale

    if readingfromfile:
      for field in distancefields:
        object.__setattr__(self, field.name, field.type(power=powers[field.name], pscale=pscale, **{field.metadata["pixelsormicrons"](self): getattr(self, field.name)}))
