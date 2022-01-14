import abc, argparse, collections, contextlib, itertools, more_itertools, re, sys

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

class ArgParseAddToDictBase(argparse.Action, abc.ABC):
  def __init__(self, *args, case_sensitive=True, type=None, key_type=None, value_type=None, **kwargs):
    self.case_sensitive = case_sensitive
    self.key_type = key_type
    self.value_type = value_type
    if type is not None: raise TypeError("Use key_type and value_type instead of type")
    if not case_sensitive and key_type is not None: raise TypeError("case_sensitive=False and key_type are incompatible")
    super().__init__(*args, **kwargs)
  @abc.abstractmethod
  def process_values(self, values):
    values = list(values)
    if not self.case_sensitive:
      values[0] = values[0].lower()
    elif self.key_type is not None:
      values[0] = self.key_type(values[0])
    if self.value_type is not None:
      values[1:] = [self.value_type(_) for _ in values[1:]]
    return values
  def __call__(self, parser, namespace, values, option_string=None):
    k, v = self.process_values(values)
    dct = getattr(namespace, self.dest)
    if dct is None: dct = self.default; setattr(namespace, self.dest, dct)
    dct[k] = v

class ArgParseAddToDict(ArgParseAddToDictBase):
  def process_values(self, values):
    k, v = super().process_values(values)
    return k, v

class ArgParseAddTupleToDict(ArgParseAddToDictBase):
  def process_values(self, values):
    k, *v = super().process_values(values)
    return k, v

class ArgParseAddRegexToDict(ArgParseAddToDict):
  def process_values(self, values):
    k, v = super().process_values(values)
    v = re.compile(v)
    return k, v
