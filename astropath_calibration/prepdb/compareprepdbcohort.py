import contextlib, dataclassy, datetime, more_itertools, numbers, numpy as np, pathlib, tempfile
from ..baseclasses.csvclasses import Annotation, Batch, Constant, ROIGlobals, QPTiffCsv, Vertex, Region
from ..baseclasses.overlap import Overlap
from ..baseclasses.rectangle import Rectangle
from ..utilities import units
from ..utilities.tableio import readtable
from .prepdbcohort import PrepdbCohort

def assertAlmostEqual(a, b, **kwargs):
  if isinstance(a, np.ndarray) and not a.shape: a = a[()]
  if isinstance(b, np.ndarray) and not b.shape: b = b[()]
  if isinstance(a, units.safe.Distance):
    return units.np.testing.assert_allclose(a, b, **kwargs)
  elif isinstance(a, numbers.Number):
    if isinstance(b, units.safe.Distance): b = float(b)
    return np.testing.assert_allclose(a, b, **kwargs)
  elif dataclassy.functions.is_dataclass(type(a)) and type(a) == type(b):
    try:
      for field in dataclassy.fields(type(a)):
        assertAlmostEqual(getattr(a, field), getattr(b, field), **kwargs)
    except AssertionError:
      np.testing.assert_equal(a, b)
  else:
    return np.testing.assert_equal(a, b)

class ComparePrepdbCohort(PrepdbCohort):
  def runsample(self, sample, **kwargs):
    super().runsample(sample, **kwargs)
    for csv, cls, extrakwargs in (
      ("annotations", Annotation, {"pscale": sample.pscale, "apscale": sample.apscale}),
      ("batch", Batch, {}),
      ("constants", Constant, {"pscale": sample.pscale, "apscale": sample.apscale, "qpscale": sample.qpscale, "readingfromfile": True}),
      ("globals", ROIGlobals, {"pscale": sample.pscale}),
      ("rect", Rectangle, {"pscale": sample.pscale}),
      ("overlap", Overlap, {"pscale": sample.pscale, "nclip": sample.nclip, "rectangles": sample.rectangles}),
      ("qptiff", QPTiffCsv, {"pscale": sample.pscale}),
      ("vertices", Vertex, {"apscale": sample.apscale}),
      ("regions", Region, {"apscale": sample.apscale, "pscale": sample.pscale}),
    ):
      sample.logger.debug("comparing "+csv)
      filename = sample.csv(csv)
      reffilename = self.root/filename.relative_to(self.dbloadroot)
      if not filename.exists() and not reffilename.exists(): continue
      try:
        with contextlib.ExitStack() as stack:
          if csv == "vertices":
            tempfolder = pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory()))
            tempfilename = tempfolder/reffilename.name
            with open(reffilename) as f, open(tempfilename, "w") as newf:
              for i, line in enumerate(f):
                if i == 0 and "wx" not in line: break
                line = ",".join(line.split(",")[:-2]) + "\n"
                newf.write(line)
              else:
                reffilename = tempfilename
          rows = readtable(filename, cls, extrakwargs=extrakwargs, checkorder=True, fieldsizelimit=int(1e6))
          targetrows = readtable(reffilename, cls, extrakwargs=extrakwargs, checkorder=True, fieldsizelimit=int(1e6))
          for i, (row, target) in enumerate(more_itertools.zip_equal(rows, targetrows)):
            if cls is Constant and row.name == "flayers" and target.unit == "pixels": target.unit = ""
            if cls is Rectangle:
              if datetime.datetime(2018, 3, 11, 2) <= target.t <= datetime.datetime(2018, 11, 4, 2) or datetime.datetime(2019, 3, 10, 2) <= target.t <= datetime.datetime(2019, 11, 3, 2):
                target.t += datetime.timedelta(hours=4)
              else:
                target.t += datetime.timedelta(hours=5)
              if abs(row.t - target.t) <= datetime.timedelta(minutes=1): target.t = row.t
            if cls is Region: target.poly = None
            if cls is Annotation:
              target.name = target.name.lower()
            if cls is Batch:
              target.SampleID = sample.SampleID
            if cls is Overlap:
              if row.tag != target.tag and row.tag % 2 == 1 and target.tag % 2 == 0:
                if row.x1 < row.x2 and row.y1 < row.y2:
                  target.tag = 1
                elif row.x1 > row.x2 and row.y1 < row.y2:
                  target.tag = 3
                elif row.x1 < row.x2 and row.y1 > row.y2:
                  target.tag = 7
                elif row.x1 > row.x2 and row.y1 > row.y2:
                  target.tag = 9
            assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
      except:
        raise ValueError(f"Error in {filename}")

def main(args=None):
  ComparePrepdbCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
