import abc, methodtools, numpy as np

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

class ThingWithPscale(abc.ABC):
  @property
  @abc.abstractmethod
  def pscale(self): return self.__pscale
  @pscale.setter
  def pscale(self, pscale): object.__setattr__(self, "_ThingWithPscale__pscale", pscale)
  @methodtools.lru_cache()
  @property
  def onepixel(self):
    return onepixel(pscale=self.pscale)
  @methodtools.lru_cache()
  @property
  def onemicron(self):
    return onemicron(pscale=self.pscale)

class ThingWithQpscale(abc.ABC):
  @property
  @abc.abstractmethod
  def qpscale(self): return self.__qpscale
  @qpscale.setter
  def qpscale(self, qpscale): object.__setattr__(self, "_ThingWithQpscale__qpscale", qpscale)
  @methodtools.lru_cache()
  @property
  def oneqppixel(self):
    return onepixel(pscale=self.qpscale)
  @methodtools.lru_cache()
  @property
  def oneqpmicron(self):
    return onemicron(pscale=self.qpscale)

class ThingWithApscale(abc.ABC):
  @property
  @abc.abstractmethod
  def apscale(self): return self.__apscale
  @apscale.setter
  def apscale(self, apscale): object.__setattr__(self, "_ThingWithApscale__apscale", apscale)
  @methodtools.lru_cache()
  @property
  def oneappixel(self):
    return onepixel(pscale=self.apscale)
  @methodtools.lru_cache()
  @property
  def oneapmicron(self):
    return onemicron(pscale=self.apscale)

class ThingWithImscale(abc.ABC):
  @property
  @abc.abstractmethod
  def imscale(self): pass
  @imscale.setter
  def imscale(self, imscale): object.__setattr__(self, "_ThingWithImscale__imscale", imscale)
  @property
  def oneimpixel(self): return onepixel(pscale=self.imscale)
  @property
  def oneimmicron(self): return onemicron(pscale=self.imscale)

