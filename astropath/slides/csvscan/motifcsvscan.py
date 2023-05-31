import pathlib
from ...shared.argumentparser import ArgumentParserWithVersionRequirement
from ...shared.astropath_logging import printlogger, ThingWithLogger
from ...utilities import units
from ...utilities.dataclasses import MyDataClass
from ...utilities.tableio import pathfield, writetable
from ..geomcell.motifgeomcell import MotifSampleWithFields

class LoadFile(MyDataClass):
  fileid: str
  filename: pathlib.Path = pathfield()
  tablename: str
  nrows: int
  nrowsloaded: int

class MotifCsvScan(MotifSampleWithFields, ArgumentParserWithVersionRequirement, ThingWithLogger, units.ThingWithPscale):
  def __init__(self, *, csvfolder, logfolder, geomfolder, dbloadfolder, tifffolder, **kwargs):
    super().__init__(**kwargs)
    self.csvfolder = pathlib.Path(csvfolder)
    self.logfolder = pathlib.Path(logfolder)
    self.geomfolder = pathlib.Path(geomfolder)
    self.dbloadfolder = pathlib.Path(dbloadfolder)
    self.tifffolder = pathlib.Path(tifffolder)

  @property
  def logger(self):
    return printlogger("motifcsvscan")

  def run(self, **kwargs):
    csvs = [
      *(
        self.csvfolder/field.tifffile.with_suffix(".csv").name.replace(",", ".").replace("_binary_seg_maps", "")
        for field in self.fields
      ), *(
        self.geomfolder/field.tifffile.with_suffix(".csv").name.replace(",", ".")
        for field in self.fields
      ),
      self.dbloadfolder/"constants.csv",
    ]
    loadfiles = []
    for csv in csvs:
      self.logger.debug(f"Processing {csv}")
      if not csv.exists():
        raise ValueError(f"{csv} does not exist")
      if csv.name == "constants.csv":
        tablename = "Constants"
      elif csv.parent.name == "csv":
        tablename = "LoadCell"
      elif csv.parent.name == "geom":
        tablename = "CellGeom"
      else:
        assert False, csv
      with open(csv) as f:
        for nrows, line in enumerate(f):
          pass
      loadfiles.append(
        LoadFile(
          fileid="",
          filename=csv,
          tablename=tablename,
          nrows=nrows,
          nrowsloaded=0,
        )
      )

    writetable(self.dbloadfolder/"loadFiles.csv", loadfiles, header=False)


  @classmethod
  def runfromargsdicts(cls, *, initkwargs, runkwargs, misckwargs):
    with units.setup_context(misckwargs.pop("units")):
      if misckwargs:
        raise TypeError(f"Some miscellaneous kwargs were not processed:\n{misckwargs}")
      sample = cls(**initkwargs)
      with sample:
        sample.run(**runkwargs)
      return sample

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "tifffolder": parsed_args_dict.pop("tiff_folder"),
      "csvfolder": parsed_args_dict.pop("csv_folder"),
      "geomfolder": parsed_args_dict.pop("geom_folder"),
      "dbloadfolder": parsed_args_dict.pop("dbload_folder"),
      "logfolder": parsed_args_dict.pop("log_folder"),
    }

  @classmethod
  def misckwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().misckwargsfromargumentparser(parsed_args_dict),
      "units": parsed_args_dict.pop("units"),
    }

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--tiff-folder", type=pathlib.Path, required=True, help="Folder with the segmented tiffs")
    p.add_argument("--csv-folder", type=pathlib.Path, required=True, help="Folder with the phenotyped cell csvs")
    p.add_argument("--geom-folder", type=pathlib.Path, required=True, help="Folder for the geom csvs")
    p.add_argument("--dbload-folder", type=pathlib.Path, required=True, help="Folder with the dbload csvs")
    p.add_argument("--log-folder", type=pathlib.Path, required=True, help="Folder for the log files")
    p.add_argument("--units", choices=("safe", "fast", "fast_pixels", "fast_microns"), default="safe")
    return p

def main(args=None):
  return MotifCsvScan.runfromargumentparser(args=args)

if __name__ == "__main__":
  main()
