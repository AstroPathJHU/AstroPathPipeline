import collections, contextlib, cv2, itertools, matplotlib.pyplot as plt, more_itertools, numba as nb, numpy as np, os, pathlib, PIL.Image, re, scipy.stats, subprocess, sys, uncertainties as unc
if sys.platform != "cygwin": import psutil

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
  return __floattoint(flt, atol, rtol)

def weightedaverage(a, *args, **kwargs):
  from . import units
  return np.average(units.nominal_values(a), weights=1/units.std_devs(a)**2, *args, **kwargs)

def weightedvariance(a, *, subtractaverage=True):
  from . import units
  if subtractaverage:
    average = weightedaverage(a)
    if not isinstance(a, np.ndarray): a = np.array(a)
    a -= average
  return np.average(units.nominal_values(a)**2, weights=1/units.std_devs(a)**2)

def weightedstd(*args, **kwargs):
  return weightedvariance(*args, **kwargs) ** 0.5

#small helper function to crop white border out of an image
def crop_and_overwrite_image(im_path,border=0.03) :
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

#parser callback function to split a string of comma-separated values into a list
def split_csv_to_list(value) :
  return value.split(',')

#parser callback function to split a string of comma-separated values into a list of integers
def split_csv_to_list_of_ints(value) :
  try :
      return [int(v) for v in value.split(',')]
  except ValueError :
      raise ValueError(f'Option value {value} is expected to be a comma-separated list of integers!')

#parser callback function to split a string of comma-separated values into a list of floats
def split_csv_to_list_of_floats(value) :
  try :
      return [float(v) for v in value.split(',')]
  except ValueError :
      raise ValueError(f'Option value {value} is expected to be a comma-separated list of floats!')

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

from .dataclasses import MyDataClass

#helper dataclass for some common metadata information
class MetadataSummary(MyDataClass):
  slideID         : str
  project         : int
  cohort          : int
  microscope_name : str
  mindate         : str
  maxdate         : str

#A small dataclass to hold entries in the background threshold datatable
class ThresholdTableEntry(MyDataClass) :
  layer_n                 : int
  counts_threshold        : int
  counts_per_ms_threshold : float

#helper function to return a list of rectangle ns for all rectangles on the edge of the tissue for this slide
def getAlignSampleTissueEdgeRectNs(aset) :
  #get the list of sets of rectangle IDs by island
  slide_islands = aset.islands()
  edge_rect_ns = []
  #for each island
  for island in slide_islands :
    island_rects = [r for r in aset.rectangles if r.n in island]
    #get the width and height of the rectangles
    #rw, rh = island_rects[0].w, island_rects[0].h
    #get the x/y positions of the rectangles in the island
    x_poss = sorted(list(set([r.x for r in island_rects])))
    y_poss = sorted(list(set([r.y for r in island_rects])))
    #make a list of the ns of the rectangles on the edge of this island
    island_edge_rect_ns = []
    #iterate over them first from top to bottom to add the vertical edges
    for row_y in y_poss :
      row_rects = [r for r in island_rects if r.y==row_y]
      row_x_poss = sorted([r.x for r in row_rects])
      #add the rectangles of the ends
      island_edge_rect_ns+=[r.n for r in row_rects if r.x in (row_x_poss[0],row_x_poss[-1])]
      ##add any rectangles that have a gaps between them and the previous
      #for irxp in range(1,len(row_x_poss)) :
      #  if abs(row_x_poss[irxp]-row_x_poss[irxp-1])>rw :
      #    island_edge_rect_ns+=[r.n for r in row_rects if r.x in (row_x_poss[irxp-1],row_x_poss[irxp])]
    #iterate over them again from left to right to add the horizontal edges
    for col_x in x_poss :
      col_rects = [r for r in island_rects if r.x==col_x]
      col_y_poss = sorted([r.y for r in col_rects])
      #add the rectangles of the ends
      island_edge_rect_ns+=[r.n for r in col_rects if r.y in (col_y_poss[0],col_y_poss[-1])]
      ##add any rectangles that have a gaps between them and the previous
      #for icyp in range(1,len(col_y_poss)) :
      #  if abs(col_y_poss[icyp]-col_y_poss[icyp-1])>rh :
      #    island_edge_rect_ns+=[r.n for r in col_rects if r.y in (col_y_poss[icyp-1],col_y_poss[icyp])]
    #add this island's edge rectangles' ns to the total list
    edge_rect_ns+=island_edge_rect_ns
  return list(set(edge_rect_ns))

#helper function to mutate an argument parser for some very generic options
def addCommonArgumentsToParser(parser,positional_args=True,et_correction=True,flatfielding=True,warping=True) :
  #positional arguments
  if positional_args :
    parser.add_argument('rawfile_top_dir',  help='Path to the directory containing the "[slideID]/*.Data.dat" files')
    parser.add_argument('root_dir',         help='Path to the Clinical_Specimen directory with info for the given slide')
    parser.add_argument('workingdir',       help='Path to the working directory (will be created if necessary)')
  #mutually exclusive group for how to handle the exposure time correction
  if et_correction :
    et_correction_group = parser.add_mutually_exclusive_group(required=True)
    et_correction_group.add_argument('--exposure_time_offset_file',
                                     help="""Path to the .csv file specifying layer-dependent exposure time correction offsets for the slides in question
                                    [use this argument to apply corrections for differences in image exposure time]""")
    et_correction_group.add_argument('--skip_exposure_time_correction', action='store_true',
                                     help='Add this flag to entirely skip correcting image flux for exposure time differences')
  #mutually exclusive group for how to handle the flatfielding
  if flatfielding :
    flatfield_group = parser.add_mutually_exclusive_group(required=True)
    flatfield_group.add_argument('--flatfield_file',
                                 help='Path to the flatfield.bin file that should be applied to the files in this slide')
    flatfield_group.add_argument('--skip_flatfielding', action='store_true',
                                 help='Add this flag to entirely skip flatfield corrections')
  #mutually exclusive group for how to handle the warping corrections
  if warping :
    warping_group = parser.add_mutually_exclusive_group(required=True)
    warping_group.add_argument('--warp_def',   
                               help="""Path to the weighted average fit result file of the warp to apply, 
                                    or to the directory with the warp's dx and dy shift fields""")
    warping_group.add_argument('--skip_warping', action='store_true',
                               help='Add this flag to entirely skip warping corrections')

class PILmaximagepixels(contextlib.AbstractContextManager):
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
  for p, r in patternsandrepls:
    string = re.sub(p, r, string, **kwargs)
  return string

class UnequalDictsError(ValueError):
    def __init__(self, details=None):
        msg = 'Dicts have different keys'
        if details is not None:
            msg += (': index 0 has keys {}; index {} has keys {}').format(
                *details
            )

        super().__init__(msg)

def dict_zip_equal(*dicts):
  for i, d in enumerate(dicts[1:]):
    if i == 0:
      keys = dicts[i].keys()
      continue
    if d.keys() != keys:
      raise UnequalDictsError(details=(keys, i, d.keys()))
  return {k: tuple(d[k] for d in dicts) for k in keys}

def dict_product(dct):
  keys = dct.keys()
  valuelists = dct.values()
  for values in itertools.product(*valuelists):
    yield {k: v for k, v in more_itertools.zip_equal(keys, values)}

def is_relative_to(path1, path2):
  if sys.version_info >= (3, 9):
    return path1.is_relative_to(path2)
  try:
    path1.relative_to(path2)
    return True
  except ValueError:
    return False

def commonroot(*paths, __niter=0):
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
  with open(filename, newline="") as f:
    contents = f.read()
    if re.search(r"(?<!\r)\n", contents):
      raise ValueError(rf"{filename} uses unix newlines (contains \n without preceding \r)")
    if re.search(r"\r\r", contents):
      raise ValueError(rf"{filename} has messed up newlines (contains double carriage return")

def pathtomountedpath(filename):
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
  if filename.startswith("//"):
    try:
      return pathlib.WindowsPath(filename)
    except NotImplementedError:
      return pathlib.PureWindowsPath(filename)
  else:
    return guesspathtype(filename)
