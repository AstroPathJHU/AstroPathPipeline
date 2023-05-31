import contextlib, csv, numpy as np, os, pathlib, re, shutil, subprocess, sys
if sys.platform != "cygwin": import psutil

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

def rm_missing_ok(path):
  if sys.version_info >= (3, 8):
    return path.unlink(missing_ok=True)
  else:
    try:
      return path.unlink()
    except FileNotFoundError:
      pass

def rmtree_missing_ok(path, **kwargs):
  try:
    shutil.rmtree(path, **kwargs)
  except FileNotFoundError:
    pass

def rmdir_missing_ok(path, **kwargs):
  try:
    path.rmdir(**kwargs)
  except FileNotFoundError:
    pass

def iterdir_missing_ok(path, **kwargs):
  try:
    yield from path.iterdir(**kwargs)
  except FileNotFoundError:
    return

def is_relative_to(path1, path2):
  """
  Like pathlib.PurePath.is_relative_to but backported to older python versions
  """
  if sys.version_info >= (3, 9):
    return path1.is_relative_to(path2)
  try:
    path1.relative_to(path2)
    return True
  except ValueError:
    return False

def with_stem(path, stem):
  """
  Like pathlib.PurePath.stem but backported to older python versions
  """
  if sys.version_info >= (3, 9):
    return path.with_stem(stem)
  return path.with_name(stem+path.suffix)

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

def pathtomountedpath(filename):
  """
  Convert a path location to the mounted path location, if the filesystem is mounted
  """
  if sys.platform == "cygwin":
    #please note that the AstroPath framework is NOT tested on cygwin
    return pathlib.PureWindowsPath(subprocess.check_output(["cygpath", "-w", filename]).strip().decode("utf-8"))

  if not filename.is_absolute():
    try:
      return pathlib.WindowsPath(filename)
    except NotImplementedError:
      return pathlib.PureWindowsPath(filename)

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

  if not filename.is_absolute(): return filename

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
  elif "/" not in path and "\\" not in path:
    return pathlib.Path(path)
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

  regex = r"([^/:]+):/srv/[^/]*/([^/]*)"
  match = re.match(regex, filename)
  if match:
    newfilename = rf"\\{match.group(1)}\{match.group(2)}"
    try:
      return pathlib.WindowsPath(newfilename)
    except NotImplementedError:
      return pathlib.PureWindowsPath(newfilename)

  return guesspathtype(filename)

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

class CorruptMemmapError(IOError):
  def __init__(self, filename, *args, **kwargs):
    if hasattr(filename, "name"): filename = filename.name
    super().__init__(f"Failed to create memmap from corrupted file {filename}", *args, **kwargs)

@contextlib.contextmanager
def memmapcontext(filename, *args, **kwargs):
  """
  Context manager for a numpy memmap that closes the memmap
  on exit.
  """
  try:
    memmap = np.memmap(filename, *args, **kwargs)
  except OSError as e:
    if getattr(e, "winerror", None) == 8:
      raise CorruptMemmapError(filename)
    else:
      raise
  except ValueError as e:
    if str(e) == "mmap length is greater than file size":
      raise CorruptMemmapError(filename)
    else:
      raise
  try:
    yield memmap
  finally:
    memmap._mmap.close()

@contextlib.contextmanager
def field_size_limit_context(limit):
  if limit is None: yield; return
  oldlimit = csv.field_size_limit()
  try:
    csv.field_size_limit(limit)
    yield
  finally:
    csv.field_size_limit(oldlimit)

class PathGlobExists:
  def __init__(self, folder, glob, *, regex=None):
    self.folder = pathlib.Path(folder)
    self.glob = glob
    self.regex = regex
  def exists(self):
    for _ in self.folder.glob(self.glob):
      if self.regex is None or re.match(self.regex, _.name):
        return True
    return False
  def __str__(self):
    result = str(self.folder/self.glob)
    if self.regex is not None:
      result += " matching " + self.regex
    return result
  def __repr__(self):
    return "{type(self).__name__}({self.folder!r}, {self.glob!r}, regex={self.regex!r})"
