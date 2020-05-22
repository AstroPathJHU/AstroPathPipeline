import contextlib, dataclasses, matplotlib.pyplot as plt, numpy as np, os, uncertainties as unc, scipy.stats

def covariance_matrix(*args, **kwargs):
  result = np.array(unc.covariance_matrix(*args, **kwargs))
  return (result + result.T) / 2

def pullhist(array, *, binning=None, verbose=True, label="", stdinlabel=True, quantileforstats=1, **kwargs):
  pulls = np.array([_.n / _.s for _ in array], dtype=float)
  quantiles = np.array(sorted(((1-quantileforstats)/2, (1+quantileforstats)/2)))
  minpull, maxpull = np.quantile(pulls, quantiles)
  outliers = len(pulls[(minpull > pulls) | (pulls > maxpull)])
  pulls = pulls[(minpull <= pulls) & (pulls <= maxpull)]

  if stdinlabel:
    if label: label += ": "
    label += rf"$\text{{std dev}} = {np.std(pulls):.02f}$"
  plt.hist(pulls, bins=binning, label=label, **kwargs)
  if verbose:
    print(f"mean of middle {100*quantileforstats}%:   ", unc.ufloat(np.mean(pulls), scipy.stats.sem(pulls)))
    print(f"std dev of middle {100*quantileforstats}%:", unc.ufloat(np.std(pulls), np.std(pulls) / np.sqrt(2*len(pulls)-2)))
    print("n outliers: ", outliers)

@contextlib.contextmanager
def cd(dir):
  cdminus = os.getcwd()
  try:
    yield os.chdir(dir)
  finally:
    os.chdir(cdminus)

class dataclass_dc_init:
  """
  Let's say that you need a dataclass that modifies its arguments to init
  for example
  class MyDataClass:
    x: float
    y: float
    #...others
    def __init__(self, xy, **kwargs):
      #want to do
      self.__dc_init__(x=xy[0], y=xy[1], **kwargs)

  this decorator lets you do that.  the normal __init__ from the dataclass
  will be saved as __dc_init__
  """

  def __new__(thiscls, decoratedcls=None, **kwargs):
    if decoratedcls is None: return super().__new__(thiscls)
    if kwargs: raise TypeError("Can't call this with both decoratedcls and kwargs")
    return thiscls()(decoratedcls)

  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def __call__(self, cls):
    __my_init__ = cls.__init__
    del cls.__init__
    cls = dataclasses.dataclass(cls, **self.kwargs)
    cls.__dc_init__ = cls.__init__
    cls.__dc_init__.__name__ = "__dc_init__"
    cls.__init__ = __my_init__
    return cls

def floattoint(flt):
  result = int(flt)
  if result == flt: return result
  raise ValueError(f"{flt} is not an int")

from . import units

def weightedaverage(a, *args, **kwargs):
  return np.average(units.nominal_values(a), weights=1/units.std_devs(a)**2, *args, **kwargs)

def weightedvariance(a, *, subtractaverage=True):
  if subtractaverage:
    average = weightedaverage(a)
    if not isinstance(a, np.ndarray): a = np.array(a)
    a -= average
  return np.average(units.nominal_values(a)**2, weights=1/units.std_devs(a)**2)

def weightedstd(*args, **kwargs):
  return weightedvariance(*args, **kwargs) ** 0.5

@contextlib.contextmanager
def PILmaximagepixels(pixels):
  import PIL.Image
  bkp = PIL.Image.MAX_IMAGE_PIXELS
  try:
    PIL.Image.MAX_IMAGE_PIXELS = pixels
    yield
  finally:
    PIL.Image.MAX_IMAGE_PIXELS = bkp
