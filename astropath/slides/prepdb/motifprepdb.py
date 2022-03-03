import methodtools, pathlib, tifffile
from ...shared.argumentparser import ArgumentParserWithVersionRequirement
from ...shared.csvclasses import Constant
from ...shared.logging import printlogger, ThingWithLogger
from ...shared.qptiff import QPTiff
from ...utilities import units
from ...utilities.tableio import writetable

class MotifPrepDb(ArgumentParserWithVersionRequirement, ThingWithLogger, units.ThingWithPscale, units.ThingWithApscale, units.ThingWithQpscale):
  def __init__(self, *, qptifffile, dbloadfolder, tifffolder, logfolder, **kwargs):
    super().__init__(**kwargs)
    self.qptifffile = pathlib.Path(qptifffile)
    self.dbloadfolder = pathlib.Path(dbloadfolder)
    self.tifffolder = pathlib.Path(tifffolder)
    self.logfolder = pathlib.Path(logfolder)

  @methodtools.lru_cache()
  @property
  def qptiffinfo(self):
    with QPTiff(self.qptifffile) as qptiff:
      return (
        qptiff.apscale,
        None,#qptiff.qpscale,
        qptiff.position,
        len(qptiff.zoomlevels[0])
      )
  @property
  def pscale(self):
    return self.qptiffinfo[0]
  @property
  def apscale(self):
    return self.qptiffinfo[0]
  @property
  def qpscale(self):
    return self.qptiffinfo[1]
  @property
  def qptiffposition(self):
    return self.qptiffinfo[2]
  @property
  def flayers(self):
    return self.qptiffinfo[3]
  @methodtools.lru_cache()
  @property
  def HPFsize(self):
    for filename in self.tifffolder.glob("*.tif"):
      with tifffile.TiffFile(filename) as tiff:
        for page in tiff.pages:
          height, width = units.distances(pixels=page.shape, pscale=self.pscale)
          return width, height
  @property
  def fwidth(self): return self.HPFsize[0]
  @property
  def fheight(self): return self.HPFsize[1]

  def getconstants(self):
    pscales = {name: getattr(self, name) for name in ("pscale", "qpscale", "apscale")}
    constants = [
      Constant(
        name='fwidth',
        value=self.fwidth,
        **pscales,
      ),
      Constant(
        name='fheight',
        value=self.fheight,
        **pscales,
      ),
      Constant(
        name='flayers',
        value=self.flayers,
        **pscales,
      ),
      Constant(
        name='xposition',
        value=self.qptiffposition[0],
        **pscales,
      ),
      Constant(
        name='yposition',
        value=self.qptiffposition[1],
        **pscales,
      ),
      #Constant(
      #  name='qpscale',
      #  value=self.qpscale,
      #  **pscales,
      #),
      Constant(
        name='apscale',
        value=self.apscale,
        **pscales,
      ),
      Constant(
        name='pscale',
        value=self.pscale,
        **pscales,
      ),
    ]
    return constants

  @property
  def logger(self):
    return printlogger("motifprepdb")

  def csv(self, csv):
    return self.dbloadfolder/f"{csv}.csv"

  def writecsv(self, csv, *args, **kwargs):
    return writetable(self.csv(csv), *args, logger=self.logger, **kwargs)

  def prepdb(self):
    self.writecsv("constants", self.getconstants())
     
  def run(self, **kwargs):
    return self.prepdb(**kwargs)

  @classmethod
  def runfromargsdicts(cls, *, initkwargs, runkwargs, misckwargs):
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
      "qptifffile": parsed_args_dict.pop("qptiff"),
      "dbloadfolder": parsed_args_dict.pop("dbload_folder"),
      "tifffolder": parsed_args_dict.pop("tiff_folder"),
      "logfolder": parsed_args_dict.pop("log_folder"),
    }

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--qptiff", type=pathlib.Path, required=True, help="The qptiff file")
    p.add_argument("--dbload-folder", type=pathlib.Path, required=True, help="Folder for the dbload output")
    p.add_argument("--tiff-folder", type=pathlib.Path, required=True, help="Folder where the tiff files are stored")
    p.add_argument("--log-folder", type=pathlib.Path, required=True, help="Folder for the log files")
    return p

def main(args=None):
  return MotifPrepDb.runfromargumentparser(args=args)

if __name__ == "__main__":
  main()
