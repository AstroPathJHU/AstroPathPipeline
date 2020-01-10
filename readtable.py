import collections, csv

def readtable(filename, rowname, **columntypes):
  """
  Read a csv table into a list of named tuples

  filename:    csv file to read from
  rowname:     name of the class that will represent each row
  columntypes: type (or function) to be called on each element in
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
    Row = collections.namedtuple(rowname, reader.fieldnames)

    for row in reader:
      for column, typ in columntypes.items():
        row[column] = typ(row[column])

      result.append(Row(**row))

  return result
