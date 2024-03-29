import abc, os, pathlib, re

from ...hpfs.flatfield.config import CONST as FF_CONST
from ...utilities.config import CONST as UNIV_CONST
from ...shared.argumentparser import ArgumentParserWithVersionRequirement, InitAndRunFromArgumentParserBase, RunFromArgumentParserBase
from ...shared.csvclasses import Annotation, AnnotationInfo, Batch, Constant, ExposureTime, PhenotypedCell, QPTiffCsv, Region, ROIGlobals
from ...shared.rectangle import GeomLoadRectangle, PhenotypedRectangle, Rectangle
from ...shared.overlap import Overlap
from ...shared.sample import CellPhenotypeSampleBase, GeomSampleBase, ReadRectanglesDbload, TissueSampleBase, WorkflowSample
from ...shared.workflowdependency import ThingWithRoots, ThingWithWorkflowKwargs
from ...utilities.dataclasses import MyDataClass
from ...utilities.tableio import pathfield, TableReader
from ..align.alignsample import AlignSample
from ..align.field import Field, FieldBoundary, FieldOverlap
from ..align.imagestats import ImageStats
from ..align.overlap import AlignmentResult
from ..align.stitch import AffineEntry
from ..annotationinfo.annotationinfo import CopyAnnotationInfoSampleBase
from ..annowarp.annowarpsample import AnnoWarpAlignmentResult, AnnoWarpSampleInformTissueMask, WarpedVertex
from ..annowarp.stitch import AnnoWarpStitchResultEntry
from ..geomcell.geomcellsample import CellGeomLoad, GeomCellSampleDeepCell, GeomCellSampleInform, GeomCellSampleMesmer

class CsvScanRectangle(GeomLoadRectangle, PhenotypedRectangle):
  pass

class CsvScanBase(TableReader, ThingWithWorkflowKwargs):
  def __init__(self, *args, skipannotations=False, skipcells=False, segmentationalgorithms=None, **kwargs):
    self.__skipannotations = skipannotations
    self.__skipcells = skipcells
    if not segmentationalgorithms: segmentationalgorithms = ["inform"]
    if "inform" not in segmentationalgorithms:
      raise ValueError("Have to include inform segmentation")
      #this is so that the hasanycells() function below works
      #can modify it to open up the segmented component tiff if there's
      #no inform and then this restriction wouldn't be necessary
    self.__segmentationalgorithms = segmentationalgorithms
    super().__init__(*args, **kwargs)

  @property
  def segmentationalgorithms(self): return self.__segmentationalgorithms
  @property
  def skipcells(self): return self.__skipcells
  @property
  def skipannotations(self): return self.__skipannotations

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

  @property
  def workflowkwargs(self) :
    return {
      **super().workflowkwargs,
      "segmentationalgorithms": self.segmentationalgorithms,
      "skipcells": self.skipcells,
      "skipannotations": self.skipannotations,
    }

class RunCsvScanBase(CsvScanBase, ArgumentParserWithVersionRequirement, InitAndRunFromArgumentParserBase, RunFromArgumentParserBase, ThingWithRoots, ThingWithWorkflowKwargs):
  possiblesegmentationalgorithms = "inform", "deepcell", "mesmer"

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--skip-check", action="store_false", dest="checkcsvs", help="do not check the validity of the csvs")
    p.add_argument("--skip-cells", action="store_true", help="skip cells csvs (segmentation and phenotype)")
    p.add_argument("--skip-annotations", action="store_true", help="skip annotation csvs")
    p.add_argument("--ignore-csvs", action="append", type=re.compile, help="ignore extraneous csv files that match this regex", default=[])
    p.add_argument("--segmentation-algorithm", action="append", choices=cls.possiblesegmentationalgorithms, help="load cell geometry csvs from these segmentation algorithms", metavar="algorithm", dest="segmentation_algorithms")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "segmentationalgorithms": parsed_args_dict.pop("segmentation_algorithms"),
      "skipcells": parsed_args_dict.pop("skip_cells"),
      "skipannotations": parsed_args_dict.pop("skip_annotations"),
    }
    return kwargs

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "checkcsvs": parsed_args_dict.pop("checkcsvs"),
      "ignorecsvs": parsed_args_dict.pop("ignore_csvs"),
    }
    return kwargs

class CsvScanSample(WorkflowSample, ReadRectanglesDbload, GeomSampleBase, CellPhenotypeSampleBase, TissueSampleBase, RunCsvScanBase):
  rectangletype = CsvScanRectangle
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

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
      )
    }
    if not self.skipannotations:
      expectcsvs |= {
        self.csv(_) for _ in (
          "annotationinfo",
          "annotations",
          "annowarp",
          "annowarp-stitch",
          "regions",
          "vertices",
        )
      }
    if not self.skipcells:
      expectcsvs |= {
        r.geomloadcsv(algo) for r in self.rectangles for algo in self.segmentationalgorithms
      }
      expectcsvs |= {
        r.phenotypecsv for r in self.rectangles if self.hasanycells(r, "inform")
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
    annotationinfocsvs = {xml.with_suffix(".annotationinfo.csv") for xml in self.scanfolder.glob("*annotations*polygons*.xml")}
    optionalcsvs = {
      self.csv(_) for _ in (
        "globals",
        "tumorGeometry",
      )
    } | {
      r.phenotypecsv for r in self.rectangles if not self.hasanycells(r, "inform")
    } | {
      r.phenotypeQAQCcsv for r in self.rectangles
    } | annotationinfocsvs | meanimagecsvs
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
          "annotationinfo": (AnnotationInfo, "AnnotationInfo"),
          "annotations": (Annotation, "Annotations"),
          "annowarp": (AnnoWarpAlignmentResult, "AnnoWarp"),
          "annowarp-stitch": (AnnoWarpStitchResultEntry, "AnnoWarpStitch"),
          "batch": (Batch, "Batch"),
          "constants": (Constant, "Constants"),
          "exposures": (ExposureTime, "Exposures"),
          "fieldoverlaps": (FieldOverlap, "FieldOverlaps"),
          "fieldGeometry": (FieldBoundary, "FieldGeometry"),
          "fields": (Field, "Fields"),
          "globals": (ROIGlobals, "Globals"),
          "imstat": (ImageStats, "Imstat"),
          "overlap": (Overlap, "Overlap"),
          "qptiff": (QPTiffCsv, "Qptiff"),
          "rect": (Rectangle, "Rect"),
          "regions": (Region, "Regions"),
          "tumorGeometry": (FieldBoundary, "TumorGeometry"),
          "vertices": (WarpedVertex, "Vertices"),
        }[match.group(1)]
        if self.skipannotations:
          allannotationinfos = allannotations = None
        else:
          allannotationinfos = self.readcsv("annotationinfo", AnnotationInfo, extrakwargs={"scanfolder": self.scanfolder})
          allannotations = self.readcsv("annotations", Annotation, extrakwargs={"annotationinfos": allannotationinfos})
        allrectangles = self.readcsv("rect", Rectangle)
        extrakwargs = {
          "annotationinfo": {"scanfolder": self.scanfolder},
          "annotations": {"annotationinfos": allannotationinfos},
          "annowarp": {"tilesize": 0, "bigtilesize": 0, "bigtileoffset": 0},
          "fieldoverlaps": {"nclip": 8, "rectangles": allrectangles},
          "overlap": {"nclip": 8, "rectangles": allrectangles},
          "regions": {"annotations": allannotations},
          "vertices": {"annotations": allannotations},
        }.get(match.group(1), {})
        fieldsizelimit = {
          "regions": 500000,
          "tumorGeometry": 500000,
        }.get(match.group(1), None)
      elif csv.parent.parent == self.geomfolder:
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
      elif csv in meanimagecsvs | annotationinfocsvs:
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
    yield from super().inputfiles(**kwargs)
    if not self.skipcells:
      for r in self.rectangles:
        for algo in self.segmentationalgorithms:
          yield r.geomloadcsv(algo)
        if self.hasanycells(r, "inform"):
          yield r.phenotypecsv

  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    segmentationalgorithms = kwargs["segmentationalgorithms"]
    skipcells = kwargs["skipcells"]
    skipannotations = kwargs["skipannotations"]
    result = super().workflowdependencyclasses(**kwargs)
    result += [
      AlignSample,
    ]
    if not skipannotations:
      result += [
        AnnoWarpSampleInformTissueMask,
        CopyAnnotationInfoSampleBase,
      ]
    if not skipcells:
      result += [
        {
          "inform": GeomCellSampleInform,
          "deepcell": GeomCellSampleDeepCell,
          "mesmer": GeomCellSampleMesmer,
        }[algo] for algo in segmentationalgorithms
      ]
    return result

  def run(self, *args, **kwargs): return self.runcsvscan(*args, **kwargs)

  @staticmethod
  def hasanycells(rectangle, algorithm):
    try:
      with open(rectangle.geomloadcsv(algorithm)) as f:
        next(f)
        next(f)
    except (FileNotFoundError, StopIteration):
      return False
    else:
      return True

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
