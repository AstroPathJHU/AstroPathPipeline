import itertools, re

from ...baseclasses.csvclasses import Annotation, Batch, Constant, ExposureTime, PhenotypedCell, QPTiffCsv, Region, ROIGlobals, Vertex
from ...baseclasses.rectangle import GeomLoadRectangle, PhenotypedRectangle, Rectangle
from ...baseclasses.overlap import Overlap
from ...baseclasses.sample import CellPhenotypeSampleBase, GeomSampleBase, ReadRectanglesDbload, WorkflowSample
from ...utilities.dataclasses import MyDataClass
from ...utilities.tableio import pathfield
from ..align.field import Field, FieldOverlap
from ..align.imagestats import ImageStats
from ..align.overlap import AlignmentResult
from ..align.stitch import AffineEntry
from ..annowarp.annowarpsample import AnnoWarpAlignmentResult, AnnoWarpSampleInformTissueMask
from ..annowarp.stitch import AnnoWarpStitchResultEntry
from ..geom.geomsample import Boundary, GeomSample
from ..geomcell.geomcellsample import CellGeomLoad, GeomCellSample

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
        "annowarp",
        "annowarp-stitch",
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
    optionalcsvs = {
      self.csv(_) for _ in (
        "globals",
      )
    }
    unknowncsvs = set()
    folders = {self.mainfolder, self.dbload.parent, self.geomfolder.parent, self.phenotypefolder.parent.parent}
    for csv in itertools.chain(*(folder.rglob("*.csv") for folder in folders)):
      if csv == self.csv("loadfiles"):
        continue

      try:
        expectcsvs.remove(csv)
      except KeyError:
        try:
          optionalcsvs.remove(csv)
        except KeyError:
          unknowncsvs.add(csv)
          continue

      if csv.parent == self.dbload:
        match = re.match(f"{self.SlideID}_(.*)[.]csv$", csv.name)
        csvclass, tablename = {
          "affine": (AffineEntry, "Affine"),
          "align": (AlignmentResult, "Align"),
          "annotations": (Annotation, "Annotations"),
          "annowarp": (AnnoWarpAlignmentResult, "AnnoWarp"),
          "annowarp-stitch": (AnnoWarpStitchResultEntry, "AnnoWarpStitch"),
          "batch": (Batch, "Batch"),
          "constants": (Constant, "Constants"),
          "exposures": (ExposureTime, "Exposures"),
          "fieldoverlaps": (FieldOverlap, "FieldOverlaps"),
          "fieldGeometry": (Boundary, "FieldGeometry"),
          "fields": (Field, "Fields"),
          "globals": (ROIGlobals, "Globals"),
          "imstat": (ImageStats, "Imstat"),
          "overlap": (Overlap, "Overlap"),
          "qptiff": (QPTiffCsv, "Qptiff"),
          "rect": (Rectangle, "Rect"),
          "regions": (Region, "Regions"),
          "tumorGeometry": (Boundary, "TumorGeometry"),
          "vertices": (Vertex, "Vertices"),
        }[match.group(1)]
        extrakwargs = {
          #"annowarp": {"tilesize": 100*self.oneimpixel, "bigtilesize": np.array([1400, 2100])*self.oneimpixel, "bigtileoffset": }
          "annowarp": {"tilesize": 0, "bigtilesize": 0, "bigtileoffset": 0, "imscale": 1},
          "fieldoverlaps": {"nclip": 8, "rectangles": self.rectangles},
        }.get(match.group(1), {})
      elif csv.parent == self.geomfolder:
        csvclass = CellGeomLoad
        tablename = "CellGeom"
        extrakwargs = {}
      elif csv.parent == self.phenotypetablesfolder:
        csvclass = PhenotypedCell
        tablename = "Cell"
        extrakwargs = {}
      else:
        assert False

      toload.append((csv, csvclass, tablename, extrakwargs))

    if expectcsvs or unknowncsvs:
      errors = []
      if expectcsvs:
        errors.append("Some csvs are missing: "+", ".join(str(_) for _ in sorted(expectcsvs)))
      if unknowncsvs:
        errors.append("Unknown csvs: "+", ".join(str(_) for _ in sorted(unknowncsvs)))
      raise ValueError("\n".join(errors))

    loadfiles = [self.processcsv(*args) for args in toload]

    self.writecsv("loadfiles", loadfiles)

  def processcsv(self, csv, csvclass, tablename, extrakwargs={}):
    self.logger.debug(f"Processing {csv}")
    #read the csv, to check that it's valid
    rows = self.readtable(csv, csvclass, extrakwargs=extrakwargs)
    nrows = len(rows)
    return LoadFile(
      fileid="",
      SlideID=self.SlideID,
      filename=csv,
      tablename=tablename,
      nrows=nrows,
      nrowsloaded=0,
    )

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, **otherworkflowkwargs):
    dbload = dbloadroot/SlideID/"dbload"
    return dbload/f"{SlideID}_loadfiles.csv"

  @property
  def inputfiles(self): return []

  @classmethod
  def logmodule(cls): return "csvscan"

  @classmethod
  def workflowdependencies(cls):
    return [AnnoWarpSampleInformTissueMask, GeomCellSample, GeomSample] + super().workflowdependencies()


class LoadFile(MyDataClass):
  fileid: str
  SlideID: str
  filename: pathfield()
  tablename: str
  nrows: int
  nrowsloaded: int
