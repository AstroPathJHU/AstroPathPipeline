import abc, dataclasses
from .core import Distance

def distancefield(pixelsormicrons, *, metadata={}, power=1, dtype=float, **kwargs):
  kwargs["metadata"] = {
    "writefunction": {"pixels": lambda x: x.pixels, "microns": lambda x: x.microns}[pixelsormicrons],
    "readfunction": dtype,
    "isdistancefield": True,
    "pixelsormicrons": pixelsormicrons,
    "power": power,
    **metadata,
  }
  return dataclasses.field(**kwargs)

@dataclasses.dataclass
class DataClassWithDistances(abc.ABC):
  @abc.abstractproperty
  def pixelsormicrons(self): pass

  def __post_init__(self, pscale):
    object.__setattr__(self, "pscale", pscale)

    distancefields = [field for field in dataclasses.fields(type(self)) if field.metadata.get("isdistancefield", False)]
    for field in distancefields:
      if field.metadata["pixelsormicrons"] != self.pixelsormicrons:
        raise ValueError(f"{type(self)} takes {self.pixelsormicrons}, but {field.name} is expecting {field.metadata['pixelsormicrons']}")

    distances = [getattr(self, _.name) for _ in distancefields]
    distances = [_ for _ in distances if _]

    usedistances = {isinstance(_, Distance) for _ in distances}
    if len(usedistances) > 1:
      raise ValueError(f"Provided some distances and some pixels to {type(self).__name__} - this is dangerous!")
    usedistances = usedistances.pop()

    pscale = {pscale} if pscale is not None else set()
    if usedistances:
      pscale |= {_.pscale for _ in distances}
    if not pscale:
      raise TypeError("Have to either provide pscale explicitly or give coordinates in units.Distance form")
    if len(pscale) > 1:
      raise units.UnitsError(f"Provided inconsistent pscales {pscale}")
    pscale = pscale.pop()

    if not usedistances:
      for field in distancefields:
        object.__setattr__(self, field.name, Distance(power=field.metadata["power"], pscale=pscale, **{self.pixelsormicrons: getattr(self, field.name)}))
