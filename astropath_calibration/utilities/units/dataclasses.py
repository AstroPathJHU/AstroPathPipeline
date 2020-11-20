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

def distancefield(pixelsormicrons, *, metadata={}, power=1, dtype=float, secondfunction=None, **kwargs):
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

  kwargs["metadata"] = {
    "writefunction": lambda *args, pixelsormicrons, **kwargs: secondfunction({
      "pixels": pixels,
      "microns": microns,
    }[pixelsormicrons](*args, **kwargs)),
    "readfunction": dtype,
    "isdistancefield": True,
    "pixelsormicrons": pixelsormicrons,
    "power": power,
    "writefunctionkwargs": lambda object: {"pscale": object.pscale, "power": power(object), "pixelsormicrons": pixelsormicrons(object)},
    **metadata,
  }
  return dataclasses.field(**kwargs)

@dataclasses.dataclass
class DataClassWithDistances(ThingWithPscale):
  @classmethod
  def distancefields(cls):
    return [field for field in dataclasses.fields(cls) if field.metadata.get("isdistancefield", False)]

  def _distances_passed_to_init(self):
    return [getattr(self, _.name) for _ in self.distancefields()]

  def __post_init__(self, pscale, readingfromfile=False):
    distancefields = self.distancefields()

    powers = {}
    for field in distancefields:
      power = field.metadata["power"]
      if callable(power):
        power = power(self)
      if not isinstance(power, numbers.Number):
        raise TypeError(f"power should be a number or a function, not {type(power)}")
      powers[field.name] = power

    usedistances = False
    if currentmode == "safe" and any(powers.values()):
      distances = self._distances_passed_to_init()
      usedistances = {isinstance(_, safe.Distance) for _ in distances if _}
      if len(usedistances) > 1:
        raise ValueError(f"Provided some distances and some pixels/microns to {type(self).__name__} - this is dangerous!")
      if usedistances:
        usedistances = usedistances.pop()
        if usedistances and readingfromfile: assert False #shouldn't be able to happen
        if not usedistances and not readingfromfile:
          raise ValueError("Have to init with readingfromfile=True if you're not providing distances")
      else:
        usedistances = False

    pscale = {pscale}
    if usedistances:
      pscale |= set(_pscale([_ for _ in distances if _]))
    pscale.discard(None)
    if len(pscale) == 1:
      pscale, = pscale
    elif not any(powers.values()):
      pscale = None
    elif not pscale:
      raise TypeError("Have to either provide pscale explicitly or give coordinates in units.Distance form")
    elif len(pscale) > 1:
      raise UnitsError(f"Provided inconsistent pscales {pscale}")
    else:
      assert False, "This can't happen"

    object.__setattr__(self, "pscale", pscale)

    if readingfromfile:
      for field in distancefields:
        object.__setattr__(self, field.name, field.type(power=powers[field.name], pscale=pscale, **{field.metadata["pixelsormicrons"](self): getattr(self, field.name)}))
