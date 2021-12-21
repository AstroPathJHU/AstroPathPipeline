import abc, os, pathlib, re

from ...hpfs.flatfield.config import CONST as FF_CONST
from ...utilities.config import CONST as UNIV_CONST
from ...shared.argumentparser import RunFromArgumentParser
from ...shared.csvclasses import Annotation, Batch, Constant, ExposureTime, PhenotypedCell, QPTiffCsv, Region, ROIGlobals
from ...shared.rectangle import GeomLoadRectangle, PhenotypedRectangle, Rectangle
from ...shared.overlap import Overlap
from ...shared.sample import CellPhenotypeSampleBase, GeomSampleBase, ReadRectanglesDbload, WorkflowSample
from ...utilities.dataclasses import MyDataClass
from ...utilities.tableio import pathfield
from ..align.field import Field, FieldOverlap
from ..align.imagestats import ImageStats
from ..align.overlap import AlignmentResult
from ..align.stitch import AffineEntry
from ..annowarp.annowarpsample import AnnoWarpAlignmentResult, AnnoWarpSampleInformTissueMask, WarpedQPTiffVertex
from ..annowarp.mergeannotationxmls import AnnotationInfoReaderSample
from ..annowarp.stitch import AnnoWarpStitchResultEntry
from ..geom.geomsample import Boundary, GeomSample
from ..geomcell.geomcellsample import CellGeomLoad, GeomCellSample
from ...utilities.tableio import TableReader

class CsvScanRectangle(GeomLoadRectangle, PhenotypedRectangle):
  pass

class CsvScanBase(TableReader):
  @property
  @abc.abstractmethod
  def logger(self): return super().logger

  def processcsv(self, csv, csvclass, tablename, extrakwargs={}, *, SlideID, checkcsv=True, fieldsizelimit=None):
    self.logger.debug(f"Processing {csv}")
    #read the csv, to check that it's valid
    if checkcsv:
      rows = self.readtable(csv, csvclass, extrakwargs=extrakwargs, fieldsizelimit=fieldsizelimit, checknewlines=True, checkorder=True)
      nrows = len(rows)
    else:
      with open(csv) as f:
        for nrows, line in enumerate(f):
          pass
    return LoadFile(
      fileid="",
      SlideID=SlideID,
      filename=csv,
      tablename=tablename,
      nrows=nrows,
      nrowsloaded=0,
    )

  @classmethod
  def logmodule(cls): return "csvscan"

class RunCsvScanBase(CsvScanBase, RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--skip-check", action="store_false", dest="checkcsvs", help="do not check the validity of the csvs")
    p.add_argument("--ignore-csvs", action="append", type=re.compile, help="ignore extraneous csv files that match this regex", default=[])
    return p

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "checkcsvs": parsed_args_dict.pop("checkcsvs"),
      "ignorecsvs": parsed_args_dict.pop("ignore_csvs"),
    }
    return kwargs

class CsvScanSample(RunCsvScanBase, AnnotationInfoReaderSample, WorkflowSample, ReadRectanglesDbload, GeomSampleBase, CellPhenotypeSampleBase):
  rectangletype = CsvScanRectangle
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.readannotationinfo()

  @property
  def logger(self): return super().logger

  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "geomfolder": self.geomfolder,
      "phenotypefolder": self.phenotypefolder,
    }

  def processcsv(self, *args, **kwargs):
    return super().processcsv(*args, SlideID=self.SlideID, **kwargs)

  def runcsvscan(self, *, checkcsvs=True, ignorecsvs=[]):
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
    }
    expectcsvs |= {
      r.geomloadcsv for r in self.rectangles
    }

    def hasanycells(rectangle):
      try:
        with open(rectangle.geomloadcsv) as f:
          next(f)
          next(f)
      except (FileNotFoundError, StopIteration):
        return False
      else:
        return True
    expectcsvs |= {
      r.phenotypecsv for r in self.rectangles if hasanycells(r)
    }

    meanimagecsvs = {
      self.im3folder/f"{self.SlideID}-mean.csv",
      self.im3folder/UNIV_CONST.MEANIMAGE_DIRNAME/FF_CONST.FIELDS_USED_CSV_FILENAME,
      self.im3folder/UNIV_CONST.MEANIMAGE_DIRNAME/f"{self.SlideID}-{FF_CONST.BACKGROUND_THRESHOLD_CSV_FILE_NAME_STEM}",
      self.im3folder/UNIV_CONST.MEANIMAGE_DIRNAME/f"{self.SlideID}-{FF_CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}",
      self.im3folder/UNIV_CONST.MEANIMAGE_DIRNAME/f"{self.SlideID}-{FF_CONST.METADATA_SUMMARY_THRESHOLDING_IMAGES_CSV_FILENAME}",
      self.im3folder/UNIV_CONST.MEANIMAGE_DIRNAME/f"{self.SlideID}-{FF_CONST.THRESHOLDING_DATA_TABLE_CSV_FILENAME}",
      self.im3folder/UNIV_CONST.MEANIMAGE_DIRNAME/FF_CONST.IMAGE_MASKING_SUBDIR_NAME/FF_CONST.LABELLED_MASK_REGIONS_CSV_FILENAME,
    }
    optionalcsvs = {
      self.csv(_) for _ in (
        "annotationinfo",
        "globals",
      )
    } | {
      r.phenotypeQAQCcsv for r in self.rectangles if hasanycells(r)
    } | meanimagecsvs
    goodcsvs = set()
    unknowncsvs = set()
    folders = {self.mainfolder, self.dbload.parent, self.geomfolder.parent, self.phenotypefolder.parent.parent}
    for folder, csv in ((folder, csv) for folder in folders for csv in folder.rglob("*.csv")):
      if csv == self.csv("loadfiles"):
        continue

      try:
        expectcsvs.remove(csv)
      except KeyError:
        try:
          optionalcsvs.remove(csv)
        except KeyError:
          if any(otherfolder/csv.relative_to(folder) in expectcsvs|optionalcsvs|goodcsvs for otherfolder in folders): continue
          if any(regex.match(os.fspath(csv.relative_to(folder))) for regex in ignorecsvs): continue
          unknowncsvs.add(csv)
          continue

      goodcsvs.add(csv)

      if csv.parent == self.dbload:
        match = re.match(f"{self.SlideID}_(.*)[.]csv$", csv.name)
        csvclass, tablename = {
          "affine": (AffineEntry, "Affine"),
          "align": (AlignmentResult, "Align"),
          "annotations": (Annotation, "Annotations"),
          "annotationinfo": (Constant, "AnnotationInfo"),
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
          "vertices": (WarpedQPTiffVertex, "Vertices"),
        }[match.group(1)]
        allrectangles = self.readcsv("rect", Rectangle)
        extrakwargs = {
          "annowarp": {"tilesize": 0, "bigtilesize": 0, "bigtileoffset": 0},
          "fieldoverlaps": {"nclip": 8, "rectangles": allrectangles},
          "overlap": {"nclip": 8, "rectangles": allrectangles},
          "vertices": {"bigtilesize": 0, "bigtileoffset": 0}
        }.get(match.group(1), {})
        fieldsizelimit = {
          "regions": 500000,
          "tumorGeometry": 500000,
        }.get(match.group(1), None)
      elif csv.parent == self.geomfolder:
        csvclass = CellGeomLoad
        tablename = "CellGeom"
        extrakwargs = {}
        fieldsizelimit = None
      elif csv.parent == self.phenotypetablesfolder:
        csvclass = PhenotypedCell
        tablename = "Cell"
        extrakwargs = {}
        fieldsizelimit = None
      elif csv.parent == self.phenotypeQAQCtablesfolder:
        continue
      elif csv in meanimagecsvs:
        continue
      else:
        assert False, csv

      toload.append({"csv": csv, "csvclass": csvclass, "tablename": tablename, "extrakwargs": extrakwargs, "fieldsizelimit": fieldsizelimit})

    toload.sort(key=lambda x: ((x["csvclass"]==CellGeomLoad), (x["csvclass"]==PhenotypedCell), x["csv"]))

    if expectcsvs or unknowncsvs:
      errors = []
      if expectcsvs:
        errors.append("Some csvs are missing: "+", ".join(str(_) for _ in sorted(expectcsvs)))
      if unknowncsvs:
        errors.append("Unknown csvs: "+", ".join(str(_) for _ in sorted(unknowncsvs)))
      raise ValueError("\n".join(errors))

    loadfiles = [self.processcsv(checkcsv=checkcsvs, **kwargs) for kwargs in toload]

    self.writecsv("loadfiles", loadfiles, header=False)

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, **otherworkflowkwargs):
    dbload = dbloadroot/SlideID/UNIV_CONST.DBLOAD_DIR_NAME
    return [dbload/f"{SlideID}_loadfiles.csv"]

  def inputfiles(self, **kwargs):
    return super().inputfiles(**kwargs)

  @classmethod
  def workflowdependencyclasses(cls):
    return [AnnoWarpSampleInformTissueMask, GeomCellSample, GeomSample] + super().workflowdependencyclasses()

  run = runcsvscan

class LoadFile(MyDataClass):
  fileid: str
  SlideID: str
  filename: pathlib.Path = pathfield()
  tablename: str
  nrows: int
  nrowsloaded: int

def main(args=None):
  CsvScanSample.runfromargumentparser(args)

if __name__ == "__main__":
  main()
