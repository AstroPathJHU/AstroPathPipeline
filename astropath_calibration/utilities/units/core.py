import numpy as np

class UnitsError(Exception): pass

@np.vectorize
def convertpscale(distance, oldpscale, newpscale, power):
  from . import Distance, microns
  return Distance(microns=microns(distance, pscale=oldpscale, power=power), pscale=newpscale, power=power)


