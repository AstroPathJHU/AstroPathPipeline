import csv, dataclasses, logging

logger = logging.getLogger("align")

def readtable(filename, rownameorclass, **columntypes):
  """
  Read a csv table into a list of named tuples

  filename:       csv file to read from
  rownameorclass: class that will represent each row, to be called
                  with **kwargs with the keywords based on the column
                  headers.  Alternatively you can give a name, and a
                  dataclass will be automatically created with that name.
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
  with open(filename) as f:
    reader = csv.DictReader(f)
    if isinstance(rownameorclass, str):
      Row = dataclasses.make_dataclass(
        rownameorclass,
        [
          (fieldname, columntypes.get(fieldname, str))
          for fieldname in reader.fieldnames
        ]
      )
    else:
      Row = rownameorclass
      for field in dataclasses.fields(Row):
        if field.name not in reader.fieldnames:
          continue
          #hopefully it has a default value!
          #otherwise we will get an error when
          #reading the first row
        typ = field.metadata.get("readfunction", field.type)
        if field.name in columntypes:
          if columntypes[field.name] != typ:
            raise TypeError(
              f"The type for {field.name} in your dataclass {Row.__name__} "
              f"and the type provided in readtable are inconsistent "
              f"({typ} != {columntypes[field.name]})"
            )
        else:
          columntypes[field.name] = typ

    for row in reader:
      for column, typ in columntypes.items():
        row[column] = typ(row[column])

      result.append(Row(**row))

  return result

def writetable(filename, rows, *, rowclass=None, retry=False, printevery=float("inf")):
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

  fieldnames = [field.name for field in dataclasses.fields(rowclass)]

  try:
    with open(filename, "w") as f:
      writer = csv.DictWriter(f, fieldnames, lineterminator='\n')
      writer.writeheader()
      for i, row in enumerate(rows, start=1):
        if printevery is not None and not i % printevery:
          logger.info(f"{i} / {size}")
        writer.writerow(dataclasses.asdict(row))
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
  if not dataclasses._is_dataclass_instance(obj):
    raise TypeError("asrow() should be called on dataclass instances")

  result = []
  for f in fields(obj):
    if not f.metadata.get("includeintable", True): continue
    value = _asdict_inner(getattr(obj, f.name), dict_factory)
    value = f.metadata.get("writefunction", lambda x: x)(value)
    result.append((f.name, value))

  return dict_factory(result)
