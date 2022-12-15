import abc, contextlib, pathlib
from .argumentparser import ArgumentParserWithVersionRequirement
from .logging import printlogger, ThingWithLogger
from ..utilities import units

class TenXSampleBase(ArgumentParserWithVersionRequirement, ThingWithLogger, units.ThingWithPscale, contextlib.ExitStack):
  def __init__(self, *, mainfolder, SlideID, **kwargs):
    super().__init__(**kwargs)
    self.__mainfolder = pathlib.Path(mainfolder)

  @property
  def mainfolder(self): return self.__mainfolder
  @property
  def wholeslidefolder(self): return self.mainfolder/"whole_slide"
  @property
  def deepzoomfolder(self): return self.wholeslidefolder/"deepzoom"
  @property
  def tilefolder(self): return self.mainfolder/"tile"
  @property
  def csvfolder(self): return self.tilefolder/"csv"
  @property
  def pngfolder(self): return self.tilefolder/"nuclear_mask"
  @property
  def geomfolder(self): return self.tilefolder/"geom"

  @property
  def wsitiff(self):
    result, = self.wholeslidefolder.glob("*.tif")
    return result
  @property
  def pscale(self): return 1

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
      "mainfolder": parsed_args_dict.pop("main_folder"),
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
    p.add_argument("main_folder", type=pathlib.Path, help="Folder with the whole_slide and tile folders")
    p.add_argument("--units", choices=("safe", "fast", "fast_pixels", "fast_microns"), default="safe")
    return p

  @property
  @abc.abstractmethod
  def logmodule(self): pass
  @property
  def logger(self): return printlogger(self.logmodule)
