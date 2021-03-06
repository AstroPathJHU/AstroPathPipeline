import contextlib, csv, dataclasses, dataclassy, pathlib

from .dataclasses import MetaDataAnnotation, MyDataClass
from .misc import dummylogger
from .units.dataclasses import DataClassWithDistances

def readtable(filename, rownameorclass, *, extrakwargs={}, fieldsizelimit=None, filter=lambda row: True, checkorder=False, maxrows=float("inf"), **columntypes):
  """
  Read a csv table into a list of named tuples

  filename:       csv file to read from
  rownameorclass: class that will represent each row, to be called
                  with **kwargs with the keywords based on the column
                  headers.  Alternatively you can give a name, and a
                  dataclass will be automatically created with that name.
  extrakwargs:    will be passed to the the class that creates each row
  columntypes:    type (or function) to be called on each element in
                  that column.  Default is it's just left as a string.

  Example:
    table.csv contains
      ID,x,y
      A,1,3
      B,2,4.5

    If you call
      table = readtable("table.csv", "Point", x=float, y=float)
    you will get
      [
        Point(ID="A", x=1.0, y=3.0),
        Point(ID="B", x=2.0, y=4.5),
      ]
    where Point is a class created specially for this table.

    You can access the column values through table[0].ID (= "A")
  """

  result = []
  with field_size_limit_context(fieldsizelimit), open(filename) as f:
    reader = csv.DictReader(f)
    if isinstance(rownameorclass, str):
      Row = dataclassy.make_dataclass(
        rownameorclass,
        [
          (fieldname, columntypes.get(fieldname, str))
          for fieldname in reader.fieldnames
        ]
      )
    else:
      Row = rownameorclass
      if not issubclass(Row, MyDataClass):
        raise TypeError(f"{Row} should inherit from {MyDataClass}")
      if checkorder:
        columnnames = list(reader.fieldnames)
        fieldnames = [field for field in dataclassy.fields(Row) if field in columnnames]
        if fieldnames != columnnames:
          raise ValueError(f"Column names and dataclass field names are not in the same order\n{columnnames}\n{fieldnames}")
      for field, fieldtype in dataclassy.fields(Row).items():
        if field not in reader.fieldnames:
          continue
          #hopefully it has a default value!
          #otherwise we will get an error when
          #reading the first row
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

    if issubclass(Row, DataClassWithDistances) and "readingfromfile" not in extrakwargs:
      extrakwargs["readingfromfile"] = True

    for i, row in enumerate(reader):
      if i >= maxrows: break
      for column, typ in columntypes.items():
        row[column] = typ(row[column])

      if not filter(row): continue

      result.append(Row(**row, **extrakwargs))

  return result

def writetable(filename, rows, *, rowclass=None, retry=False, printevery=float("inf"), logger=dummylogger):
  """
  Write a csv table into filename based on the rows.
  The rows should all be the same dataclass type.
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

  try:
    with open(filename, "w") as f:
      writer = csv.DictWriter(f, fieldnames, lineterminator='\n')
      writer.writeheader()
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

@contextlib.contextmanager
def field_size_limit_context(limit):
  if limit is None: yield; return
  oldlimit = csv.field_size_limit()
  try:
    csv.field_size_limit(limit)
    yield
  finally:
    csv.field_size_limit(oldlimit)

def pathfield(**metadata):
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

  metadata = {
    "readfunction": guesspathtype,
    **metadata,
  }

  return MetaDataAnnotation(pathlib.Path, **metadata)
