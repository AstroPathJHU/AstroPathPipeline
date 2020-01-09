import collections, csv

def readtable(filename, **columntypes):
	result = []
	with open(filename) as f:
		reader = csv.DictReader(f)
		Row = collections.namedtuple("Row", reader.fieldnames)
		for row in reader:
			for column, typ in columntypes.items():
				row[column] = typ(row[column])
			result.append(Row(**row))
	return result