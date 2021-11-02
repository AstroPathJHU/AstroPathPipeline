import collections, contextlib, csv, cv2, itertools, matplotlib.pyplot as plt, more_itertools, numba as nb, numpy as np, os, pathlib, PIL.Image, re, scipy.stats, subprocess, sys, uncertainties as unc, uncertainties.umath as umath
import reikna as rk
if sys.platform != "cygwin": import psutil

def covariance_matrix(*args, **kwargs):
  """
  Covariance matrix for the uncertainties module
  that enforces symmetry (the normal one is symmetric up
  to rounding errors)
  """
  result = np.array(unc.covariance_matrix(*args, **kwargs))
  return (result + result.T) / 2

def pullhist(array, *, binning=None, verbose=True, label="", stdinlabel=True, quantileforstats=1, **kwargs):
  """
  Make a histogram of uncertainties.nominal_values(array) / uncertainties.std_dev(array)
  """
  pulls = np.array([_.n / _.s for _ in array], dtype=float)
  quantiles = np.array(sorted(((1-quantileforstats)/2, (1+quantileforstats)/2)))
  minpull, maxpull = np.quantile(pulls, quantiles)
  outliers = len(pulls[(minpull > pulls) | (pulls > maxpull)])
  pulls = pulls[(minpull <= pulls) & (pulls <= maxpull)]

  if stdinlabel:
    if label: label += ": "
    label += rf"$\text{{std dev}} = {np.std(pulls):.02f}$"
  if verbose:
    print(f"mean of middle {100*quantileforstats}%:   ", unc.ufloat(np.mean(pulls), scipy.stats.sem(pulls)))
    print(f"std dev of middle {100*quantileforstats}%:", unc.ufloat(np.std(pulls), np.std(pulls) / np.sqrt(2*len(pulls)-2)))
    print("n outliers: ", outliers)
  return plt.hist(pulls, bins=binning, label=label, **kwargs)

@contextlib.contextmanager
def cd(dir):
  """
  Change the current working directory to a different directory,
  and go back when leaving the context manager.
  """
  cdminus = os.getcwd()
  try:
    yield os.chdir(dir)
  finally:
    os.chdir(cdminus)

def crop_and_overwrite_image(im_path,border=0.03) :
  """
  small helper function to crop white border out of an image
  """
  im = cv2.imread(im_path)
  y_border = int(im.shape[0]*(border/2))
  x_border = int(im.shape[1]*(border/2))
  min_y = 0; max_y = im.shape[0]
  min_x = 0; max_x = im.shape[1]
  while np.min(im[min_y:min_y+y_border,:,:])==255 :
      min_y+=1
  while np.min(im[max_y-y_border:max_y,:,:])==255 :
      max_y-=1
  while np.min(im[:,min_x:min_x+x_border,:])==255 :
      min_x+=1
  while np.min(im[:,max_x-x_border:max_x,:])==255 :
      max_x-=1
  cv2.imwrite(im_path,im[min_y:max_y+1,min_x:max_x+1,:])

def save_figure_in_dir(pyplot_inst,figname,save_dirpath=None) :
  """
  Save the current figure in the given pyplot instance with a given name and crop it. 
  If save_dirpath is given the figure is saved in that directory (possibly creating it)
  """
  if save_dirpath is not None :
    if not save_dirpath.is_dir() :
      save_dirpath.mkdir(parents=True)
    with cd(save_dirpath) :
      pyplot_inst.savefig(figname)
      pyplot_inst.close()
      crop_and_overwrite_image(figname)
  else :
    pyplot_inst.savefig(figname)
    pyplot_inst.close()
    crop_and_overwrite_image(figname)

@nb.vectorize([nb.int64(nb.float64, nb.float64, nb.float64)])
def __floattoint(flt, atol, rtol):
  result = int(flt)
  flt = float(flt)
  for thing in result, result+1, result-1:
    #this version needs https://github.com/numba/numba/pull/6074 or https://github.com/numba/numba/pull/4610
    #if np.isclose(thing, flt, atol=atol, rtol=rtol):
    if np.abs(thing-flt) <= atol + rtol * np.abs(flt):
      return thing
  raise ValueError("not an int")

def floattoint(flt, *, atol=0, rtol=1e-10):
  """
  If flt is an integer within absolute and relative tolerance atol and rtol,
  return it as an int.  Otherwise raise an error.
  """
  try:
    return __floattoint(flt, atol, rtol)
  except ValueError as e:
    if str(e) == "not an int":
      raise ValueError(f"not an int: {flt}")
    raise

def weightedaverage(a, *args, **kwargs):
  """
  Weighted average of an array a, where the weights are 1/error^2.
  a should contain objects from the uncertainties module.
  """
  from . import units
  return np.average(units.nominal_values(a), weights=1/units.std_devs(a)**2, *args, **kwargs)

def weightedvariance(a, *, subtractaverage=True):
  """
  Weighted variance of an array a, where the weights are 1/error^2.
  a should contain objects from the uncertainties module.
  """
  from . import units
  if subtractaverage:
    average = weightedaverage(a)
    if not isinstance(a, np.ndarray): a = np.array(a)
    a -= average
  return np.average(units.nominal_values(a)**2, weights=1/units.std_devs(a)**2)

def weightedstd(*args, **kwargs):
  """
  Weighted standard deviation of an array a, where the weights are 1/error^2.
  a should contain objects from the uncertainties module.
  """
  return weightedvariance(*args, **kwargs) ** 0.5

def split_csv_to_list(value) :
  """
  parser callback function to split a string of comma-separated values into a list
  """
  return value.split(',')

def split_csv_to_list_of_floats(value) :
  """
  parser callback function to split a string of comma-separated values into a list of floats
  """
  try :
      return [float(v) for v in value.split(',')]
  except ValueError :
      raise ValueError(f'Option value {value} is expected to be a comma-separated list of floats!')

def dict_of_init_par_values_callback(value) :
  """
  argument parser callback to return a dictionary of fit parameter initial values
  """
  try :
    pairs = value.split()
    return_dict = {}
    for pair in pairs :
      name,value = pair.split('=')
      return_dict[name] = float(value)
    return return_dict
  except Exception :
      raise ValueError(f'Option value {value} is expected to be a space-separated string of name=float pairs!')

def dict_of_par_bounds_callback(value) :
  """
  argument parser callback to return a dictionary of fit parameter bounds
  """
  try :
    pairs = value.split()
    return_dict = {}
    for pair in pairs :
      name,bounds = pair.split('=')
      low_bound,high_bound = bounds.split(':')
      return_dict[name] = (float(low_bound),float(high_bound))
    return return_dict
  except Exception as e :
    raise ValueError(f'Option value {value} is expected to be a comma-separated list of name=low_bound:high_bound pairs! Exception: {e}')

class PILmaximagepixels(contextlib.AbstractContextManager):
  """
  Context manager to increase the maximum image pixels for PIL.
  Using this context manager will never decrease the maximum pixels.
  You can also set it to None, which is equivalent to infinity.
  """
  __maximagepixelscounter = collections.Counter()
  __defaultmaximagepixels = PIL.Image.MAX_IMAGE_PIXELS
  def __init__(self, maximagepixels):
    self.__maximagepixels = maximagepixels
  def __enter__(self):
    self.__maximagepixelscounter[self.__maximagepixels] += 1
    self.__updatemaximagepixels()
  def __exit__(self, *exc):
    self.__maximagepixelscounter[self.__maximagepixels] -= 1
    self.__updatemaximagepixels()
  @classmethod
  def __updatemaximagepixels(cls):
    elements = set(cls.__maximagepixelscounter.elements()) | {cls.__defaultmaximagepixels}
    if None in elements:
      PIL.Image.MAX_IMAGE_PIXELS = None
    else:
      PIL.Image.MAX_IMAGE_PIXELS = max(elements)

@contextlib.contextmanager
def memmapcontext(filename, *args, **kwargs):
  """
  Context manager for a numpy memmap that closes the memmap
  on exit.
  """
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

def re_subs(string, *patternsandrepls, **kwargs):
  """
  Helper function to do multiple iterations of re.sub
  in sequence.
  """
  for p, r in patternsandrepls:
    string = re.sub(p, r, string, **kwargs)
  return string

class UnequalDictsError(ValueError):
  """
  Copied from more_itertools.UnequalIterablesError,
  but for dicts.
  """
  def __init__(self, details=None):
    msg = 'Dicts have different keys'
    if details is not None:
      msg += (': index 0 has keys {}; index {} has keys {}').format(
        *details
      )

    super().__init__(msg)

def dict_zip_equal(*dicts):
  """
  Similar to more_itertools.zip_equal(), but for dicts

    >>> dict_zip_equal({1: 2, 3: 4}, {1: 3, 3: 5})
    {1: (2, 3), 3: (4, 5)}
    >>> dict_zip_equal({1: 2}, {1: 2, 3: 4})
    Traceback (most recent call last):
      ...
    astropath.utilities.misc.UnequalDictsError: Dicts have different keys: index 0 has keys dict_keys([1]); index 1 has keys dict_keys([1, 3])
  """
  for i, d in enumerate(dicts):
    if i == 0:
      keys = dicts[i].keys()
      continue
    if d.keys() != keys:
      raise UnequalDictsError(details=(keys, i, d.keys()))
  return {k: tuple(d[k] for d in dicts) for k in keys}

def dict_product(dct):
  """
  like itertools.product, but the input and outputs are dicts.

    >>> for _ in dict_product({1: (2, 3), 4: (5, 6)}): print(_)
    {1: 2, 4: 5}
    {1: 2, 4: 6}
    {1: 3, 4: 5}
    {1: 3, 4: 6}
  """
  keys = dct.keys()
  valuelists = dct.values()
  for values in itertools.product(*valuelists):
    yield {k: v for k, v in more_itertools.zip_equal(keys, values)}

def is_relative_to(path1, path2):
  """
  Like pathlib.Path.is_relative_to but backported to older python versions
  """
  if sys.version_info >= (3, 9):
    return path1.is_relative_to(path2)
  try:
    path1.relative_to(path2)
    return True
  except ValueError:
    return False

def commonroot(*paths, __niter=0):
  """
  Give the common root of a number of paths
    >>> paths = [pathlib.Path(_) for _ in ("/a/b/c", "/a/b/d", "/a/c")]
    >>> commonroot(*paths) == pathlib.Path("/a")
    True
  """
  assert __niter <= 100*len(paths)
  path1, *others = paths
  if not others: return path1
  path2, *others = others
  if len({path1.is_absolute(), path2.is_absolute()}) > 1:
    raise ValueError("Can't call commonroot with some absolute and some relative paths")
  if path1 == path2: return commonroot(path1, *others)
  path1, path2 = sorted((path1, path2), key=lambda x: len(x.parts))
  return commonroot(path1, path2.parent, *others, __niter=__niter+1)

def checkwindowsnewlines(filename):
  r"""
  Check that the file consistently uses windows newlines \r\n
  """
  with open(filename, newline="") as f:
    contents = f.read()
    if re.search(r"(?<!\r)\n", contents):
      raise ValueError(rf"{filename} uses unix newlines (contains \n without preceding \r)")
    if re.search(r"\r\r", contents):
      raise ValueError(rf"{filename} has messed up newlines (contains double carriage return")

def pathtomountedpath(filename):
  """
  Convert a path location to the mounted path location, if the filesystem is mounted
  """
  if sys.platform == "cygwin":
    #please note that the AstroPath framework is NOT tested on cygwin
    return pathlib.PureWindowsPath(subprocess.check_output(["cygpath", "-w", filename]).strip().decode("utf-8"))

  bestmount = bestmountpoint = None
  for mount in psutil.disk_partitions(all=True):
    mountpoint = mount.mountpoint
    mounttarget = mount.device
    if mountpoint == mounttarget: continue
    if mounttarget.startswith("auto"): continue
    mountpoint = pathlib.Path(mountpoint)
    if not is_relative_to(filename, mountpoint): continue
    if bestmount is None or is_relative_to(mountpoint, bestmountpoint):
      bestmount = mount
      bestmountpoint = mountpoint
      bestmounttarget = mounttarget

  if bestmount is None:
    return filename

  bestmounttarget = mountedpath(bestmounttarget)

  return bestmounttarget/filename.relative_to(bestmountpoint)

def mountedpathtopath(filename):
  """
  Convert a path on a mounted filesystem to the corresponding path on
  the current filesystem.
  """
  if sys.platform == "cygwin":
    #please note that the AstroPath framework is NOT tested on cygwin
    return pathlib.Path(subprocess.check_output(["cygpath", "-u", filename]).strip().decode("utf-8"))

  bestmount = bestmountexists = bestresult = None
  for mount in psutil.disk_partitions(all=True):
    mountpoint = mount.mountpoint
    mounttarget = mount.device
    if mountpoint == mounttarget: continue
    if mounttarget.startswith("auto"): continue
    if "/" not in mounttarget and "\\" not in mounttarget: continue
    mountpoint = pathlib.Path(mountpoint)
    mounttarget = mountedpath(mounttarget)
    if not is_relative_to(filename, mounttarget): continue
    result = mountpoint/filename.relative_to(mounttarget)
    mountexists = result.exists()
    if bestmount is None or mountexists and not bestmountexists:
      bestmount = mount
      bestresult = result
      bestmountexists = mountexists

  if bestmount is None:
    return filename

  return bestresult

def guesspathtype(path):
  """
  return a WindowsPath, PosixPath, PureWindowsPath, or PurePosixPath,
  as appropriate, guessing based on the types of slashes in the path
  """
  if isinstance(path, pathlib.PurePath):
    return path
  if pathlib.Path(path).exists(): return pathlib.Path(path)
  if "/" in path and "\\" not in path:
    try:
      return pathlib.PosixPath(path)
    except NotImplementedError:
      return pathlib.PurePosixPath(path)
  elif "\\" in path and "/" not in path:
    try:
      return pathlib.WindowsPath(path)
    except NotImplementedError:
      return pathlib.PureWindowsPath(path)
  else:
    raise ValueError(f"Can't guess the path type for {path}")

def mountedpath(filename):
  """
  like guesspathtype, but if the path starts with // it will assume
  it's a network path on windows
  """
  if filename.startswith("//"):
    try:
      return pathlib.WindowsPath(filename)
    except NotImplementedError:
      return pathlib.PureWindowsPath(filename)
  else:
    return guesspathtype(filename)

def sorted_eig(*args, **kwargs):
  val, vec = np.linalg.eig(*args, **kwargs)
  order = np.argsort(val)[::-1]
  return val[order], vec[:,order]

def get_GPU_thread(interactive) :
  """
  Create and return a Reikna Thread object to use for running some computations on the GPU
  interactive : if True (and some GPU is available), user will be given the option to choose a device 
  """
  api = rk.cluda.ocl_api()
  #return a thread from the API
  return api.Thread.create(interactive=interactive)

@contextlib.contextmanager
def field_size_limit_context(limit):
  if limit is None: yield; return
  oldlimit = csv.field_size_limit()
  try:
    csv.field_size_limit(limit)
    yield
  finally:
    csv.field_size_limit(oldlimit)

def vips_format_dtype(format_or_dtype):
  """
  https://libvips.github.io/pyvips/intro.html#numpy-and-pil
  """
  result = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
  }
  for k, v in list(result.items()):
    result[v] = k
    result[np.dtype(v)] = k
  return result[format_or_dtype]

def vips_image_to_array(img, *, singlelayer=True):
  """
  https://libvips.github.io/pyvips/intro.html#numpy-and-pil
  """
  shape = [img.height, img.width, img.bands]
  if singlelayer:
    if shape[-1] != 1: raise ValueError("Have to write singlelayer=False if the image has more than one channel")
    del shape[-1]
  return np.ndarray(
    buffer=img.write_to_memory(),
    dtype=vips_format_dtype(img.format),
    shape=shape,
  )

def array_to_vips_image(array):
  """
  https://libvips.github.io/pyvips/intro.html#numpy-and-pil
  """
  try:
    import pyvips
  except ImportError:
    raise ImportError("Please pip install pyvips to use this functionality")

  if len(array.shape) == 2:
    height, width = array.shape
    bands = 1
  else:
    height, width, bands = array.shape

  return pyvips.Image.new_from_memory(
    array,
    format=vips_format_dtype(array.dtype),
    width=width,
    height=height,
    bands=bands,
  )

def vips_sinh(image):
  try:
    #https://github.com/libvips/pyvips/pull/282
    return image.sinh()
  except AttributeError:
    exp = image.exp()
    minusexp = 1/exp
    return (exp - minusexp) / 2

def affinetransformation(scale=None, rotation=None, shear=None, translation=None):
  """
  https://github.com/scikit-image/scikit-image/blob/f0fcdc8d73c20d741e31e5d57efad1a38426f7fd/skimage/transform/_geometric.py#L876-L880
  """
  if scale is None: scale = 1
  if np.isscalar(scale): scale = scale, scale
  sx, sy = scale

  if rotation is None: rotation = 0
  if shear is None: shear = 0

  if translation is None: translation = 0
  if np.isscalar(translation): translation = translation, translation
  tx, ty = translation

  return np.array([
    [sx * umath.cos(rotation), -sy * umath.sin(rotation + shear), tx],
    [sx * umath.sin(rotation),  sy * umath.cos(rotation + shear), ty],
    [                       0,                                 0, 1],
  ])
