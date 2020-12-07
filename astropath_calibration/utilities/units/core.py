import methodtools, numpy as np

currentmodule = None

class UnitsError(Exception): pass

@np.vectorize
def convertpscale(distance, oldpscale, newpscale, power=1):
  from . import Distance, microns
  return Distance(microns=microns(distance, pscale=oldpscale, power=power), pscale=newpscale, power=power)

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
