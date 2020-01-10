import collections, csv

def readtable(filename, rowname, **columntypes):
  result = []
  with open(filename) as f:
    reader = csv.DictReader(f)
    Row = collections.namedtuple(rowname, reader.fieldnames)
    for row in reader:
      for column, typ in columntypes.items():
        row[column] = typ(row[column])
      result.append(Row(**row))
  return result
