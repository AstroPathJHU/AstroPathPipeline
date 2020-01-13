import csv, dataclasses

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
    or table[1][1] (= 2.0)
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
        if field.name in columntypes:
          if columntypes[field.name] != field.type:
            raise TypeError(
              f"The type for {field.name} in your dataclass {Row.__name__} "
              f"and the type provided in readtable are inconsistent "
              f"({field.type} != {columntypes[field.name]})"
            )
        else:
          columntypes[field.name] = field.type

    for row in reader:
      for column, typ in columntypes.items():
        row[column] = typ(row[column])

      result.append(Row(**row))

  return result

def writetable(filename, rows):
  """
  Write a csv table into filename based on the rows.
  The rows should all be the same dataclass type.
  """

  rowclasses = {type(_) for _ in rows}
  if len(rowclasses) > 1:
    raise TypeError(
      "Provided rows of different types:\n  "
      + "\n  ".join(_.__name__ for _ in rowclasses))
  rowclass = rowclasses.pop()
  fieldnames = [field.name for field in dataclasses.fields(rowclass)]

  with open(filename, "w") as f:
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(dataclasses.asdict(row))
