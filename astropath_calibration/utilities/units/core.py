import methodtools

class UnitsError(Exception): pass

class Distance:
  def __new__(self, *args, **kwargs):
    return currentmodule.Distance(*args, **kwargs)

class ThingWithPscale:
  @methodtools.lru_cache()
  @property
  def onepixel(self):
    return Distance(pixels=1, pscale=self.pscale)
