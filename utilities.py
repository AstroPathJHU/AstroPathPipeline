import numpy as np, uncertainties as unc

def covariance_matrix(*args, **kwargs):
  result = np.array(unc.covariance_matrix(*args, **kwargs))
  return (result + result.T) / 2
