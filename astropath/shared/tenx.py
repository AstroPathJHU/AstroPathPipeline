import abc, contextlib, json, methodtools, numpy as np, pathlib
from .argumentparser import ArgumentParserWithVersionRequirement
from .imageloader import ImageLoaderTiff
from .astropath_logging import printlogger, ThingWithLogger
from ..utilities import units
from ..utilities.tableio import boolasintfield, optionalfield
from ..utilities.units.dataclasses import DataClassWithPscale, distancefield

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
  def dbloadfolder(self): return self.wholeslidefolder/"dbload"

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
  @methodtools.lru_cache()
  @property
  def wsiloader(self):
    return ImageLoaderTiff(filename=self.wsitiff, layers=[1, 2, 3])
  def using_wsi(self, **kwargs):
    return self.wsiloader.using_image(**kwargs)
  @property
  def wsi(self):
    return self.wsiloader.image
  @property
  def pscale(self): return 1

  @property
  def metadatafolder(self):
    return self.mainfolder/"metadata"
  @property
  def spotsfile(self):
    result, = self.metadatafolder.glob("*_alignment_file.json")
    return result
  @methodtools.lru_cache()
  @property
  def __spots_and_matrix(self):
    spots = {}
    matrix = None
    with open(self.spotsfile) as f:
      for k, v in json.load(f).items():
        if k == "transform":
          matrix = np.asarray(v)
        elif k in ("serialNumber", "area", "checksum"):
          pass
        elif k in ("oligo", "fiducial"):
          spots[k] = []
          for spotkwargs in v:
            spotkwargs = {kk: vv*self.onepixel if kk in ("dia", "imageX", "imageY") else vv for kk, vv in spotkwargs.items()}
            spots[k].append(Spot(**spotkwargs, pscale=self.pscale))
        else:
          raise ValueError(k)
    return spots, matrix
  @property
  def spots(self):
    return self.__spots_and_matrix[0]
  @property
  def spotstransform(self):
    return self.__spots_and_matrix[1]

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

class Spot(DataClassWithPscale):
  x: int
  y: int
  row: int
  col: int
  dia: units.Distance = distancefield(pixelsormicrons="pixels")
  fidName: str = optionalfield(None, readfunction=str)
  imageX: units.Distance = distancefield(pixelsormicrons="pixels")
  imageY: units.Distance = distancefield(pixelsormicrons="pixels")
  tissue: bool = boolasintfield(None, optional=True)
