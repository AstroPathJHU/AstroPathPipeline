import abc, contextlib, csv, dataclasses, dataclassy, datetime

from ..shared.logging import dummylogger
from .dataclasses import MetaDataAnnotation, MyDataClass
from .misc import checkwindowsnewlines, field_size_limit_context, guesspathtype, mountedpathtopath, pathtomountedpath

def readtable(filename, rowclass, *, extrakwargs={}, fieldsizelimit=None, filter=lambda row: True, checkorder=False, checknewlines=False, maxrows=float("inf"), header=True, **columntypes):
  """
  Read a csv table into a list of named tuples

  filename:       csv file to read from
  rowclass:       class that will represent each row, to be called
                  with **kwargs with the keywords based on the column
                  headers.
  extrakwargs:    will be passed to the the class that creates each row (default: {})
  fieldsizelimit: maximum length of a field in the csv (default: keep the default from python)
  filter:         only include rows that match this filter.
                  should be a function that takes the row dict
                  (default: use all rows)
  maxrows:        only use this many rows of the csv file (default: use all rows)
  columntypes:    type (or function) to be called on each element in
                  that column.  Default is it's just left as a string.
  header:         does the csv file have a header (default: True)
  checkorder:     check that the order of columns is as expected from the dataclass
                  and raise an error if it's not (default: False)
  checknewlines:  check that the newlines in the csv file are windows format
                  and raise an error if they're not (default: False)

  Example:
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...   filename = pathlib.Path(folder)/"test.csv"
    ...   with open(filename, "w") as f:
    ...     f.write('''
    ...       ID,x,y
    ...       A,1,3
    ...       B,2,4.5
    ...     '''.replace(" ", "").strip()) and None #avoid printing return value of f.write
    ...   class Point(MyDataClass):
    ...     ID: str
    ...     x: float
    ...     y: float
    ...   table = readtable(filename, Point)
    ...   print(table)
    [Point(ID='A', x=1.0, y=3.0), Point(ID='B', x=2.0, y=4.5)]

  You can access the column values through table[0].ID (= "A")

  The row class inherits from MyDataClass and therefore can have
  fields with metadata.  readtable uses the following metadata:
    readfunction:   called on the string read from the file to give the
                    actual value of the field.  Default is the type
                    of the annotation, which works for simple cases like
                    int and float
    includeintable: if this is False, the field is not read from the file.
                    (it has to have a default value or the dataclass __init__
                    will fail)
  """

  result = []
  if checknewlines:
    checkwindowsnewlines(filename)
  with field_size_limit_context(fieldsizelimit), open(filename) as f:
    if header:
      fieldnames = None
    else:
      fieldnames = [f for f in dataclassy.fields(rowclass) if rowclass.metadata(f).get("includeintable", True)]
    reader = csv.DictReader(f, fieldnames=fieldnames)
    Row = rowclass
    if not issubclass(Row, (MyDataClass)):
      raise TypeError(f"{Row} should inherit from {MyDataClass}")
    if checkorder:
      columnnames = list(reader.fieldnames)
      fieldnames = [field for field in dataclassy.fields(Row) if field in columnnames]
      if fieldnames != columnnames:
        raise ValueError(f"Column names and dataclass field names are not in the same order\n{columnnames}\n{fieldnames}")
    for field, fieldtype in dataclassy.fields(Row).items():
      if field not in reader.fieldnames:
        continue
      typ = Row.metadata(field).get("readfunction", fieldtype)
      if field in columntypes:
        if columntypes[field] != typ:
          raise TypeError(
            f"The type for {field} in your dataclass {Row.__name__} "
            f"and the type provided in readtable are inconsistent "
            f"({typ} != {columntypes[field]})"
          )
      else:
        columntypes[field] = typ

    if "readingfromfile" not in extrakwargs:
      extrakwargs["readingfromfile"] = True
    if "extrakwargs" not in extrakwargs:
      extrakwargs["extrakwargs"] = extrakwargs

    for i, row in enumerate(reader):
      if i >= maxrows: break
      for column, typ in columntypes.items():
        row[column] = typ(row[column])

      if not filter(row): continue

      result.append(Row(**row, **extrakwargs))

  return result

def writetable(filename, rows, *, rowclass=None, retry=False, printevery=float("inf"), logger=dummylogger, header=True, append=False):
  """
  Write a csv table into filename based on the rows.
  The rows should all be the same dataclass type.

  filename:   file to write to
  rows:       list of dataclass objects, one for each row
  rowclass:   class of the rows (optional unless the rows are not
              all the same type, but are subclasses of the same class.
              in that case you can provide the common base as rowclass)
  retry:      interactively ask to try again if there's a permission error
              (usually on windows if you have the file open in excel)
  printevery: print after writing multiples of this many rows
  logger:     logger object for printevery
  header:     write the header row (default: True)

  Example:
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...   filename = pathlib.Path(folder)/"test.csv"
    ...   class Point(MyDataClass):
    ...     ID: str
    ...     x: float
    ...     y: float
    ...   input = [Point("A", 1, 3), Point("B", 2, 4.5)]
    ...   writetable(filename, input)
    ...   output = readtable(filename, Point)
    ...   input == output
    True

  The row class inherits from MyDataClass and therefore can have
  fields with metadata.  readtable uses the following metadata:
    writefunction:       called on the object in the dataclass field
                         to return a string that gets written to the file
                         (default: str)
    writefunctionkwargs: a function that takes in the dataclass object
                         and returns extra kwargs to be passed to writefunction
                         (default: lambda obj: {})
    includeintable:      if this is False, the field is not written in the table
  """
  size = len(rows)
  if printevery > size:
    printevery = None
  if printevery is not None:
    logger.info(f"writing {filename}, which will have {size} rows")

  rowclasses = {type(_) for _ in rows}
  if rowclass is None:
    if len(rowclasses) > 1:
      raise TypeError(
        "Provided rows of different types:\n  "
        + "\n  ".join(_.__name__ for _ in rowclasses)
      )
    rowclass = rowclasses.pop()
  else:
    badclasses = {cls for cls in rowclasses if not issubclass(cls, rowclass)}
    if badclasses:
      raise TypeError(f"Provided rows of types that aren't consistent with rowclass={rowclass.__name__}:\n  "
        + "\n  ".join(_.__name__ for _ in badclasses)
      )

  if not issubclass(rowclass, MyDataClass):
    raise TypeError(f"{rowclass} should inherit from {MyDataClass}")

  fieldnames = [f for f in dataclassy.fields(rowclass) if rowclass.metadata(f).get("includeintable", True)]

  if append:
    with contextlib.ExitStack() as stack:
      try:
        f = stack.enter_context(open(filename, "r"))
      except FileNotFoundError:
        append = False
      else:
        with open(filename, "r") as f:
          if header:
            readfieldnames = None
          else:
            readfieldnames = fieldnames
          reader = csv.DictReader(f, fieldnames=readfieldnames)
          if reader.fieldnames != fieldnames:
            raise ValueError(f"Inconsistent fieldnames for append:\nprevious lines were written with:\n{reader.fieldnames}\nnew lines would be written with:\n{fieldnames}")
          #this check handles two cases:
          # 1) if not header, checks that the line lengths match
          # 2) checks that a previous write didn't get interrupted
          #    and end up with a truncated line
          for row in reader:
            nfields = sum(1 for k, v in row.items() if k is not None is not v) + len(row.get(None, []))
            if nfields != len(fieldnames):
              raise ValueError(f"Inconsistent number of fields for append: previous lines had {nfields}, new lines would have {len(fieldnames)}")
  try:
    openmode = "a" if append else "w"
    with open(filename, openmode, newline='') as f:
      writer = csv.DictWriter(f, fieldnames, lineterminator='\r\n')
      if header and not append: writer.writeheader()
      for i, row in enumerate(rows, start=1):
        if printevery is not None and not i % printevery:
          logger.debug(f"{i} / {size}")
        writer.writerow(asrow(row))
  except PermissionError:
    if retry:
      result = None
      while True:
        result = input(f"Permission error writing to {filename} - do you want to retry? yes/no  ")
        if result == "yes":
          return writetable(filename, rows, retry=False, rowclass=rowclass, printevery=printevery)
        elif result == "no":
          raise
    else:
      raise
  if printevery is not None:
    logger.info("finished!")

class TableReader(abc.ABC):
  """
  Base class that has a readtable function
  so that you can override it and call super()
  """
  def readtable(self, *args, **kwargs):
    return readtable(*args, **kwargs)

def asrow(obj, *, dict_factory=dict):
  """
  loosely inspired by https://github.com/python/cpython/blob/77c623ba3d084e99d68c30f368bd7fbd7f175b60/Lib/dataclasses.py#L1052
  """
  if not isinstance(obj, MyDataClass):
    raise TypeError("asrow() should be called on MyDataClass instances")

  result = []
  for f in dataclassy.fields(obj):
    metadata = type(obj).metadata(f)
    if not metadata.get("includeintable", True): continue
    value = dataclasses._asdict_inner(getattr(obj, f), dict_factory)
    writefunction = metadata.get("writefunction", lambda x: x)
    writefunctionkwargs = metadata.get("writefunctionkwargs", lambda object: {})(obj)
    value = writefunction(value, **writefunctionkwargs)
    result.append((f, value))

  return dict_factory(result)

def pathfield(*defaultvalue, **metadata):
  """
  returns a MetaDataAnnotation for writing a path.
  if the path location is on a mount, it tries to find the actual
  path location through the mount.
  """
  metadata = {
    "readfunction": lambda x: mountedpathtopath(guesspathtype(x)),
    "writefunction": pathtomountedpath,
    **metadata,
  }

  return MetaDataAnnotation(*defaultvalue, **metadata)

def boolasintfield(*defaultvalue, **metadata):
  """
  returns a MetaDataAnnotation for writing a bool as an int (i.e. 1 or 0)
  """
  metadata = {
    "readfunction": lambda x: bool(int(x)),
    "writefunction": int,
    **metadata,
  }

  return MetaDataAnnotation(*defaultvalue, **metadata)

def datefield(dateformat, *defaultvalue, optional=False, **metadata):
  """
  returns a MetaDataAnnotation for writing a date

    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...   filename = pathlib.Path(folder)/"test.csv"
    ...   with open(filename, "w") as f:
    ...     f.write('''
    ...       x,date
    ...       1,1/1/2021 1:00:00
    ...       2,1/2/2021 2:00:00
    ...     '''.replace("      ", "").strip()) and None #avoid printing return value of f.write
    ...   class DateDataClass(MyDataClass):
    ...     x: int
    ...     date: datetime.datetime = datefield("%d/%m/%Y %H:%M:%S")
    ...   table = readtable(filename, DateDataClass)
    ...   print(table)
    [DateDataClass(x=1, date=datetime.datetime(2021, 1, 1, 1, 0)), DateDataClass(x=2, date=datetime.datetime(2021, 2, 1, 2, 0))]
  """
  metadata = {
    "readfunction": lambda x: datetime.datetime.strptime(x, dateformat),
    "writefunction": lambda x: x.strftime(format=dateformat),
    **metadata,
  }
  return (optionalfield if optional else MetaDataAnnotation)(*defaultvalue, **metadata)

def timestampfield(*, optional=False, **metadata):
  """
  returns a MetaDataAnnotation for writing a as a unix timestamp
  """
  metadata = {
    "readfunction": lambda x: datetime.datetime.fromtimestamp(int(x)),
    "writefunction": lambda x: int(datetime.datetime.timestamp(x)),
    **metadata,
  }
  return (optionalfield if optional else MetaDataAnnotation)(**metadata)

def optionalfield(readfunction, *, writefunction=str, **metadata):
  """
  returns a MetaDataAnnotation for a field that is optional
  (None <--> blank in the csv)

    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...   filename = pathlib.Path(folder)/"test.csv"
    ...   with open(filename, "w") as f:
    ...     f.write('''
    ...       x,y
    ...       1,2
    ...       2,
    ...     '''.replace("      ", "").strip()) and None #avoid printing return value of f.write
    ...   class OptionalYDataClass(MyDataClass):
    ...     x: int
    ...     y: int = optionalfield(int)
    ...   table = readtable(filename, OptionalYDataClass)
    ...   print(table)
    [OptionalYDataClass(x=1, y=2), OptionalYDataClass(x=2, y=None)]
  """
  metadata = {
    "readfunction": lambda x: None if not x else readfunction(x),
    "writefunction": lambda x: "" if x is None else writefunction(x),
    **metadata,
  }
  return MetaDataAnnotation(**metadata)
