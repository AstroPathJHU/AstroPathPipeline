import methodtools

currentmodule = None

class UnitsError(Exception): pass

class Distance:
  def __new__(self, *args, **kwargs):
    return currentmodule.Distance(*args, **kwargs)

def onepixel(pscale):
  return Distance(pixels=1, pscale=pscale)
def onemicron(pscale):
  return Distance(microns=1, pscale=pscale)

class ThingWithPscale:
  @methodtools.lru_cache()
  @property
  def onepixel(self):
    return onepixel(pscale=self.pscale)
  @methodtools.lru_cache()
  @property
  def onemicron(self):
    return onemicron(pscale=self.pscale)
