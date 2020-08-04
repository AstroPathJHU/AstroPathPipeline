import contextlib, dataclasses, fractions, logging, matplotlib.pyplot as plt, numpy as np, os, uncertainties as unc, scipy.stats, tifffile

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

@np.vectorize
def floattoint(flt, *, atol=0, rtol=0):
  result = int(flt)
  flt = float(flt)
  for thing in result, result+1, result-1:
    if np.isclose(thing, flt, atol=atol, rtol=rtol): return thing
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

#parser callback function to split a string of comma-separated values into a list
def split_csv_to_list(value) :
  return value.split(',')

#parser callback function to split a string of comma-separated values into a list of integers
def split_csv_to_list_of_ints(value) :
  try :
      return [int(v) for v in value.split(',')]
  except ValueError :
      raise ValueError(f'Option value {value} is expected to be a comma-separated list of integers!')

#parser callback function to split a string of comma-separated name=value pairs into a dictionary
def split_csv_to_dict_of_floats(value) :
  try :
    pairs = value.split(',')
    return_dict = {}
    for pair in pairs :
      name,value = pair.split('=')
      return_dict[name] = float(value)
    return return_dict
  except Exception :
      raise ValueError(f'Option value {value} is expected to be a comma-separated list of name=float pairs!')

#helper function to split a string of comma-separated name=(low bound:high bound) pairs into a dictionary
def split_csv_to_dict_of_bounds(value) :
  try :
    pairs = value.split(',')
    return_dict = {}
    for pair in pairs :
      name,bounds = pair.split('=')
      low_bound,high_bound = bounds.split(':')
      return_dict[name] = (float(low_bound),float(high_bound))
    return return_dict
  except Exception as e :
    raise ValueError(f'Option value {value} is expected to be a comma-separated list of name=low_bound:high_bound pairs! Exception: {e}')

@contextlib.contextmanager
def PILmaximagepixels(pixels):
  import PIL.Image
  bkp = PIL.Image.MAX_IMAGE_PIXELS
  try:
    PIL.Image.MAX_IMAGE_PIXELS = pixels
    yield
  finally:
    PIL.Image.MAX_IMAGE_PIXELS = bkp

@contextlib.contextmanager
def memmapcontext(filename, *args, **kwargs):
  try:
    memmap = np.memmap(filename, *args, **kwargs)
  except OSError as e:
    if hasattr(filename, "name"): filename = filename.name
    if getattr(e, "winerror", None) == 8:
      raise IOError(f"Failed to create memmap from corrupted file {filename}")
  try:
    yield memmap
  finally:
    memmap._mmap.close()

def tiffinfo(*, filename=None, page=None):
  if filename is page is None:
    raise TypeError("Have to provide either filename or page")
  with tifffile.TiffFile(filename) if filename is not None else contextlib.nullcontext() as f:
    if filename is not None:
      if page is None: page = 0
      page = f.pages[page]
    resolutionunit = page.tags["ResolutionUnit"].value
    xresolution = page.tags["XResolution"].value
    xresolution = fractions.Fraction(*xresolution)
    yresolution = page.tags["YResolution"].value
    yresolution = fractions.Fraction(*yresolution)
    if xresolution != yresolution: raise ValueError(f"x and y have different resolutions {xresolution} {yresolution}")
    resolution = float(xresolution)
    kw = {
      tifffile.TIFF.RESUNIT.CENTIMETER: "centimeters",
    }[resolutionunit]
    pscale = float(units.Distance(pixels=resolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1))
    height, width = units.distances(pixels=page.shape, pscale=pscale, power=1)

    return pscale, width, height

dummylogger = logging.getLogger("dummy")
dummylogger.addHandler(logging.NullHandler())
dummylogger.warningglobal = dummylogger.warning
