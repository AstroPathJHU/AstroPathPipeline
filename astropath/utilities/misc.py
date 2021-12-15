import argparse, collections, contextlib, itertools, more_itertools, re, sys

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

class recursionlimit(contextlib.AbstractContextManager):
  """
  Context manager to increase the recursion limit.
  Using this context manager will never decrease the recursion limit.
  """
  __recursionlimitcounter = collections.Counter()
  __defaultrecursionlimit = sys.getrecursionlimit()
  def __init__(self, recursionlimit):
    self.__recursionlimit = recursionlimit
  def __enter__(self):
    self.__recursionlimitcounter[self.__recursionlimit] += 1
    self.__updaterecursionlimit()
  def __exit__(self, *exc):
    self.__recursionlimitcounter[self.__recursionlimit] -= 1
    self.__updaterecursionlimit()
  @classmethod
  def __updaterecursionlimit(cls):
    elements = set(cls.__recursionlimitcounter.elements()) | {cls.__defaultrecursionlimit}
    sys.setrecursionlimit(max(elements))

class ArgParseAddToDict(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    k, v = values
    dct = getattr(namespace, self.dest)
    if dct is None: dct = {}; setattr(namespace, self.dest, dct)
    dct[k] = v

class ArgParseAddRegexToDict(ArgParseAddToDict):
  def __call__(self, parser, namespace, values, option_string=None):
    k, v = values
    v = re.compile(v)
    values = k, v
    return super().__call__(parser, namespace, values, option_string=None)
