import abc, argparse, pathlib, re
from ..utilities.tableio import TableReader
from ..utilities.config import CONST as UNIV_CONST
from ..utilities.misc import dict_of_init_par_values_callback, dict_of_par_bounds_callback
from .annotationpolygonxmlreader import add_rename_annotation_argument
from .workflowdependency import ThingWithRoots

class MRODebuggingMetaClass(abc.ABCMeta):
  def __new__(cls, name, bases, dct, **kwargs):
    try:
      return super().__new__(cls, name, bases, dct, **kwargs)
    except TypeError as e:
      if "Cannot create a consistent" in str(e):
        print("========================")
        print(f"MROs of bases of {name}:")
        for base in bases:
          print("------------------------")
          for c in base.__mro__:
            print(c.__name__)
        print("************************")
        print("filtered for the bad ones:")
        for base in bases:
          bad = [c for c in base.__mro__ if re.search(rf"\b{c.__name__}\b", str(e))]
          if len(bad) < 2: continue
          print("------------------------")
          print(base.__name__)
          for c in bad:
            print(c.__name__)
        print("========================")
      raise

class RunFromArgumentParserBase(ThingWithRoots, TableReader, metaclass=MRODebuggingMetaClass):
  @classmethod
  def argumentparserhelpmessage(cls):
    return cls.__doc__

  @classmethod
  def makeargumentparser(cls, *, _forworkflow=False):
    """
    Create an argument parser to run on the command line
    """
    p = argparse.ArgumentParser(description=cls.argumentparserhelpmessage())
    return p

  @classmethod
  def runfromargumentparser(cls, args=None, **kwargs):
    """
    Main function to run from command line arguments.
    This function can be called in __main__
    """
    p = cls.makeargumentparser()
    parsed_args = p.parse_args(args=args)
    return cls.runfromparsedargs(parsed_args, **kwargs)

  @classmethod
  @abc.abstractmethod
  def runfromparsedargs(cls, parsed_args):
    pass

class RunFromArgumentParser(RunFromArgumentParserBase, ThingWithRoots):
  @classmethod
  @abc.abstractmethod
  def defaultunits(cls):
    pass

  @classmethod
  def makeargumentparser(cls, **kwargs):
    """
    Create an argument parser to run on the command line
    """
    p = super().makeargumentparser(**kwargs)
    p.add_argument("root", type=pathlib.Path, help="The Clinical_Specimen folder where sample data is stored")
    p.add_argument("--units", choices=("safe", "fast", "fast_pixels", "fast_microns"), default=cls.defaultunits(), help=f"unit implementation (default: {cls.defaultunits()}; safe is only needed for debugging code)")
    p.add_argument("--im3root", type=pathlib.Path, help="root location where the sample im3 folders, containing im3 files from the microscope and some xml metadata, are stored (default: same as root)")
    p.add_argument("--informdataroot", type=pathlib.Path, help="root location where the sample inform_data folders, which contain outputs from inform, are stored (default: same as root)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--logroot", type=pathlib.Path, help="root location where the log files are stored (default: same as root)")
    g.add_argument("--no-log", action="store_true", help="do not write to log files")
    p.add_argument("--xmlfolder", type=pathlib.Path, action="append", help="additional folders to look for xml metadata", default=[], dest="xmlfolders")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--no-dev-version", help="refuse to run unless the package version is tagged", action="store_const", const="tag", dest="version_requirement")
    g.add_argument("--allow-dev-version", help="ok to run even if the package is at a dev version (default if using a log file)", action="store_const", const="commit", dest="version_requirement")
    g.add_argument("--allow-local-edits", help="ok to run even if there are local edits on top of the git commit (default if not writing to a log file)", action="store_const", const="any", dest="version_requirement")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    dct = parsed_args_dict
    initkwargs = {
      "root": dct.pop("root"),
      "im3root": dct.pop("im3root"),
      "informdataroot": dct.pop("informdataroot"),
      "logroot": dct.pop("logroot"),
      "uselogfiles": not dct.pop("no_log"),
      "xmlfolders": dct.pop("xmlfolders"),
    }
    if initkwargs["logroot"] is None: initkwargs["logroot"] = initkwargs["root"]
    if initkwargs["im3root"] is None: initkwargs["im3root"] = initkwargs["root"]
    if initkwargs["informdataroot"] is None: initkwargs["informdataroot"] = initkwargs["root"]
    return initkwargs

  @classmethod
  def misckwargsfromargumentparser(cls, parsed_args_dict):
    dct = parsed_args_dict
    misckwargs = {
      "units": dct.pop("units"),
      "version_requirement": dct.pop("version_requirement"),
    }
    return misckwargs

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    """
    Get the keyword arguments to be passed to cohort.run() from the parsed arguments
    """
    kwargs = {}
    return kwargs

  @classmethod
  def argsdictsfromargumentparser(cls, parsed_args_dict):
    """
    Get the kwargs dicts needed to run from the argparse dict
    from the parsed arguments
    """
    initkwargs = cls.initkwargsfromargumentparser(parsed_args_dict)
    misckwargs = cls.misckwargsfromargumentparser(parsed_args_dict)
    runkwargs = cls.runkwargsfromargumentparser(parsed_args_dict)

    version_requirement = misckwargs.pop("version_requirement")
    if version_requirement is None:
      version_requirement = "commit" if initkwargs["uselogfiles"] else "any"
    cls.checkversion(version_requirement)

    return {
      "initkwargs": initkwargs,
      "misckwargs": misckwargs,
      "runkwargs": runkwargs,
    }

  @staticmethod
  def checkversion(version_requirement):
    if version_requirement == "any":
      checkdate = checktag = False
    elif version_requirement == "commit":
      checkdate = True
      checktag = False
    elif version_requirement == "tag":
      checkdate = checktag = True
    else:
      assert False, version_requirement

    from ..utilities.version import env_var_no_git, astropathversionmatch

    if checkdate:
      if env_var_no_git:
        raise RuntimeError("Cannot run if environment variable _ASTROPATH_VERSION_NO_GIT is set unless you set --allow-local-edits")
      if astropathversionmatch.group("date"):
        raise ValueError("Cannot run with local edits to git unless you set --allow-local-edits")
    if checktag:
      if astropathversionmatch.group("dev"):
        raise ValueError("Specified --no-dev-version, but the current version is a dev version")

  @classmethod
  @abc.abstractmethod
  def runfromargsdicts(cls, **argsdicts): pass

  @classmethod
  def runfromparsedargs(cls, parsed_args):
    argsdict = parsed_args.__dict__.copy()
    argsdicts = cls.argsdictsfromargumentparser(argsdict)
    if argsdict:
      raise TypeError(f"Some command line arguments were not processed:\n{argsdict}")
    cls.runfromargsdicts(**argsdicts)

class Im3ArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("root2", type=pathlib.Path, help="root location of sharded im3 files")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "root2": parsed_args_dict.pop("root2"),
    }

class WorkingDirArgumentParser(RunFromArgumentParser) :
  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument('--workingdir', type=pathlib.Path,help='Path to the working directory where output should be stored')
    return p
  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    wd = parsed_args_dict.pop('workingdir')
    if wd is None :
      return super().initkwargsfromargumentparser(parsed_args_dict)
    else :
      return {
        **super().initkwargsfromargumentparser(parsed_args_dict),
        'workingdir': wd,
      }

class FileTypeArgumentParser(RunFromArgumentParser) :
  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument('--filetype',choices=['raw','flatWarp'],default='raw',
                   help=f'Whether to use "raw" files (extension {UNIV_CONST.RAW_EXT}, default) or "flatWarp" files (extension {UNIV_CONST.FLATW_EXT})')
    return p
  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      'filetype': parsed_args_dict.pop('filetype'), 
    }

class GPUArgumentParser(RunFromArgumentParser) :
  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument('--noGPU', action='store_true',
                   help='Add this flag to disable any major GPU computations')
    return p
  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      'useGPU': (not parsed_args_dict.pop('noGPU')), 
    }

class ImageCorrectionArgumentParser(RunFromArgumentParser) :
  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    g = p.add_mutually_exclusive_group()
    g.add_argument('--exposure_time_offset_file', type=pathlib.Path,
                    help='''Path to the .csv file specifying layer-dependent exposure time correction offsets for the slides in question
                    [default=None will search for a .xml file specifying dark current values]''')
    g.add_argument('--skip_exposure_time_corrections', action='store_true',
                    help='''Add this flag to skip exposure time corrections entirely''')
    p.add_argument('--flatfield_file', type=pathlib.Path,
                    help='''Path to the flatfield .bin file, or name of the file in root/Flatfield, containing the correction factors to apply 
                    [default=None skips flatfield corrections]''')
    p.add_argument('--warping_file', type=pathlib.Path,
                    help='Path to the warping summary .csv file defining the parameters of the warping pattern to apply [default=None skips warping corrections]')
    return p
  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      'et_offset_file': parsed_args_dict.pop('exposure_time_offset_file'),
      'skip_et_corrections':parsed_args_dict.pop('skip_exposure_time_corrections'),
      'flatfield_file': parsed_args_dict.pop('flatfield_file'),
      'warping_file': parsed_args_dict.pop('warping_file')
    }

class WarpFitArgumentParser(RunFromArgumentParser) :
  @classmethod
  def makeargumentparser(cls) :
    p = super().makeargumentparser()
    p.add_argument('--fixed', default=['fx','fy','p1','p2'], nargs='*',
                   help='Names of parameters to keep fixed during fitting (default = fx, fy, p1, p2)')
    p.add_argument('--init_pars', type=dict_of_init_par_values_callback, nargs='*',
                   help='Initial values for fit parameters ("parameter=value" pairs)')
    p.add_argument('--bounds', type=dict_of_par_bounds_callback, nargs='*',
                   help='Initial bounds for fit parameters ("parameter=(low_bound:high_bound)" pairs)')
    p.add_argument('--max_rad_warp', type=float, default=8.,
                   help='Maximum amount of radial warp to use for constraint')
    p.add_argument('--max_tan_warp', type=float, default=4.,
                   help='Maximum amount of tangential warp to use for constraint')
    return p
  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict) :
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      'fixed': parsed_args_dict.pop('fixed'),
      'init_pars': parsed_args_dict.pop('init_pars'),
      'bounds': parsed_args_dict.pop('bounds'),
      'max_rad_warp': parsed_args_dict.pop('max_rad_warp'),
      'max_tan_warp': parsed_args_dict.pop('max_tan_warp'),
    }

class DbloadArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--dbloadroot", type=pathlib.Path, help="root location of dbload folder (default: same as root)")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "dbloadroot": parsed_args_dict.pop("dbloadroot"),
    }

class ZoomFolderArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--zoomroot", type=pathlib.Path, required=True, help="root location of stitched wsi images")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "zoomroot": parsed_args_dict.pop("zoomroot"),
    }

class DeepZoomArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--deepzoomroot", type=pathlib.Path, required=True, help="root location of deepzoom images")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "deepzoomroot": parsed_args_dict.pop("deepzoomroot"),
    }

class MaskArgumentParser(RunFromArgumentParser):
  defaultmaskfilesuffix = ".npz"

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--maskroot", type=pathlib.Path, help="root location of mask folder (default: same as root)")
    p.add_argument("--mask-file-suffix", choices=(".npz", ".bin"), default=cls.defaultmaskfilesuffix, help=f"format for the mask files for either reading or writing (default: {cls.defaultmaskfilesuffix})")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "maskroot": parsed_args_dict.pop("maskroot"),
      "maskfilesuffix": parsed_args_dict.pop("mask_file_suffix"),
    }

class SelectRectanglesArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--selectrectangles", type=int, nargs="*", metavar="HPFID", help="select only certain HPF IDs to run on")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "selectrectangles": parsed_args_dict.pop("selectrectangles"),
    }

class SelectLayersArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--layers", type=int, nargs="*", metavar="LAYER", help="select only certain layers to run on")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "layers": parsed_args_dict.pop("layers"),
    }

class TempDirArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--temproot", type=pathlib.Path, help="root folder to save temp files in")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "temproot": parsed_args_dict.pop("temproot"),
    }

class GeomFolderArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--geomroot", type=pathlib.Path, help="root location of geom folder (default: same as root)")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "geomroot": parsed_args_dict.pop("geomroot"),
    }

class XMLPolygonReaderArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    add_rename_annotation_argument(p)
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "annotationsynonyms": parsed_args_dict.pop("annotationsynonyms"),
      "reorderannotations": parsed_args_dict.pop("reorderannotations"),
    }

class ParallelArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--njobs", type=int, help="maximum number of parallel jobs to run")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "njobs": parsed_args_dict.pop("njobs"),
    }
