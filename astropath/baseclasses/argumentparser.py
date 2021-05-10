import abc, argparse, pathlib

class RunFromArgumentParser(abc.ABC):
  @classmethod
  def argumentparserhelpmessage(cls):
    return cls.__doc__

  @classmethod
  @abc.abstractmethod
  def defaultunits(cls):
    pass

  @classmethod
  def makeargumentparser(cls):
    """
    Create an argument parser to run on the command line
    """
    p = argparse.ArgumentParser(description=cls.argumentparserhelpmessage())
    p.add_argument("root", type=pathlib.Path, help="The Clinical_Specimen folder where sample data is stored")
    p.add_argument("--units", choices=("safe", "fast", "fast_pixels", "fast_microns"), default=cls.defaultunits(), help=f"unit implementation (default: {cls.defaultunits()}; safe is only needed for debugging code)")
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
      "logroot": dct.pop("logroot"),
      "uselogfiles": not dct.pop("no_log"),
      "xmlfolders": dct.pop("xmlfolders"),
    }
    if initkwargs["logroot"] is None: initkwargs["logroot"] = initkwargs["root"]
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
  def runfromargumentparser(cls, args=None):
    """
    Main function to run from command line arguments.
    This function can be called in __main__
    """
    p = cls.makeargumentparser()
    args = p.parse_args(args=args)
    argsdict = args.__dict__.copy()
    argsdicts = cls.argsdictsfromargumentparser(argsdict)
    if argsdict:
      raise TypeError(f"Some command line arguments were not processed:\n{argsdict}")
    cls.runfromargsdicts(**argsdicts)

class Im3ArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("root2", type=pathlib.Path, help="root location of sharded im3 files")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "root2": parsed_args_dict.pop("root2"),
    }

class DbloadArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
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
  def makeargumentparser(cls):
    p = super().makeargumentparser()
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
  def makeargumentparser(cls):
    p = super().makeargumentparser()
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
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--maskroot", type=pathlib.Path, help="root location of mask folder (default: same as root)")
    p.add_argument("--mask-file-suffix", choices=(".npz", ".bin"), default=cls.defaultmaskfilesuffix, help=f"format for the mask files for either reading or writing (default: {cls.defaultmaskfilesuffix})")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "maskroot": parsed_args_dict.pop("maskroot"),
      "maskfilesuffix": parsed_args_dict.pop("maskfilesuffix"),
    }

class SelectRectanglesArgumentParser(RunFromArgumentParser):
  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
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
  def makeargumentparser(cls):
    p = super().makeargumentparser()
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
  def makeargumentparser(cls):
    p = super().makeargumentparser()
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
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument("--geomroot", type=pathlib.Path, help="root location of geom folder (default: same as root)")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "geomroot": parsed_args_dict.pop("geomroot"),
    }
