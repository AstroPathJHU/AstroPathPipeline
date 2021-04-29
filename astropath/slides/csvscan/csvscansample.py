import re

from ...baseclasses.csvclasses import Batch
from ...baseclasses.rectangle import GeomLoadRectangle, PhenotypedRectangle
from ...baseclasses.sample import CellPhenotypeSampleBase, GeomSampleBase, ReadRectanglesDbload, WorkflowSample
from ...utilities.dataclasses import MyDataClass
from ...utilities.tableio import pathfield
from ..align.stitch import AffineEntry
from ..geomcell.geomcellsample import CellGeomLoad

class CsvScanRectangle(GeomLoadRectangle, PhenotypedRectangle):
  pass

class CsvScanSample(WorkflowSample, ReadRectanglesDbload, GeomSampleBase, CellPhenotypeSampleBase):
  rectangletype = CsvScanRectangle
  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "geomfolder": self.geomfolder,
      "phenotypefolder": self.phenotypefolder,
    }

  def runcsvscan(self):
    toload = []
    expectcsvs = {
      self.csv(_) for _ in (
        "affine",
        "align",
        "annotations",
        "batch",
        "constants",
        "exposures",
        "fieldGeometry",
        "fieldoverlaps",
        "fields",
        "imstat",
        "overlap",
        "qptiff",
        "rect",
        "regions",
        "tumorGeometry",
        "vertices",
      )
    } | {
      r.geomloadcsv for r in self.rectangles
    } | {
      r.phenotypetablescsv for r in self.rectangles
    }
    folders = {self.mainfolder, self.dbloadfolder.parent, self.geomfolder.parent, self.phenotypefolder.parent.parent}
    print(folders); assert False
    for csv in itertools.chain(*(folder.rglob("*.csv") for folder in folders)):
      if csv == self.csv("loadfiles"):
        continue

      self.logger.debug(f"Processing {csv}")

      try:
        expectcsvs.remove(csv)
      except KeyError:
        self.unknowncsv(csv)

      if csv.parent == self.dbloadfolder:
        match = re.match(f"{self.SlideID}_(.*)[.]csv$", csv.name)
        csvclass = {
          "affine": AffineEntry,
        }[match.group(1)]
      elif csv.parent == self.geomfolder:
        csvclass = CellGeomLoad
      elif csv.parent == self.phenotypetablesfolder:
        csvclass = PhenotypedCell
      else:
        assert False

      toload.append((csv, csvclass))

    if expectcsvs:
      raise ValueError("Some csvs are missing: "+", ".join(str(_) for _ in sorted(expectcsvs)))

    loadfiles = [self.processcsv(csv, csvclass) for csv, csvclass in toload]

    self.writecsv("loadfiles", loadfiles)

  def processcsv(self, csv, csvclass):
    #read the csv, to check that it's valid
    rows = self.readtable(csv, csvclass)
    nrows = len(rows)
    return LoadFile(
      fileid="",
      SlideID=self.SlideID,
      filename=csv,
      classname=csvclass.csvscanname,
      nrows=nrows,
      nrowsloaded=0,
    )

class LoadFile(MyDataClass):
  fileid: str
  SlideID: str
  filename: pathfield()
  classname: str
  nrows: int
  nrowsloaded: int
