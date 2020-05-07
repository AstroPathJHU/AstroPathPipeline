import abc, dataclasses, numbers
from ..misc import floattoint
from .core import UnitsError

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

def distancefield(pixelsormicrons, *, metadata={}, power=1, dtype=float, **kwargs):
  if issubclass(dtype, numbers.Integral):
    secondfunction = floattoint
  else:
    secondfunction = lambda x: x

  kwargs["metadata"] = {
    "writefunction": {
      "pixels": lambda *args, **kwargs: secondfunction(pixels(*args, **kwargs)),
      "microns": lambda *args, **kwargs: secondfunction(microns(*args, **kwargs)),
    }[pixelsormicrons],
    "readfunction": dtype,
    "isdistancefield": True,
    "pixelsormicrons": pixelsormicrons,
    "power": power,
    "writefunctionkwargs": lambda object: {"pscale": object.pscale, "power": power},
    **metadata,
  }
  return dataclasses.field(**kwargs)

@dataclasses.dataclass
class DataClassWithDistances(abc.ABC):
  @abc.abstractproperty
  def pixelsormicrons(self): pass

  def __post_init__(self, pscale, readingfromfile=False):
    distancefields = [field for field in dataclasses.fields(type(self)) if field.metadata.get("isdistancefield", False)]
    for field in distancefields:
      if field.metadata["pixelsormicrons"] != self.pixelsormicrons:
        raise ValueError(f"{type(self)} takes {self.pixelsormicrons}, but {field.name} is expecting {field.metadata['pixelsormicrons']}")

    distances = [getattr(self, _.name) for _ in distancefields]
    distances = [_ for _ in distances if _]

    usedistances = False
    if currentmode == "safe":
      usedistances = {isinstance(_, safe.Distance) for _ in distances}
      if len(usedistances) > 1:
        raise ValueError(f"Provided some distances and some pixels/microns to {type(self).__name__} - this is dangerous!")
      usedistances = usedistances.pop()
      if usedistances and readingfromfile: assert False #shouldn't be able to happen
      if not usedistances and not readingfromfile:
        raise ValueError("Have to init with readingfromfile=True if you're not providing distances")

    pscale = {pscale}
    if usedistances:
      pscale |= set(_pscale(distances))
    pscale.discard(None)
    if not pscale:
      raise TypeError("Have to either provide pscale explicitly or give coordinates in units.Distance form")
    if len(pscale) > 1:
      raise UnitsError(f"Provided inconsistent pscales {pscale}")
    pscale = pscale.pop()

    object.__setattr__(self, "pscale", pscale)

    if readingfromfile:
      for field in distancefields:
        object.__setattr__(self, field.name, field.type(power=field.metadata["power"], pscale=pscale, **{self.pixelsormicrons: getattr(self, field.name)}))
