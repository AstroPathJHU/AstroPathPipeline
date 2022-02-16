import abc, contextlib, cv2, datetime, fractions, itertools, job_lock, jxmlease, logging, methodtools, multiprocessing as mp, numpy as np, os, pathlib, re, tempfile, tifffile, xml.etree.ElementTree as ET

from ..hpfs.flatfield.config import CONST as FF_CONST
from ..hpfs.warping.warp import CameraWarp
from ..hpfs.warping.utilities import WarpingSummary
from ..hpfs.imagecorrection.utilities import CorrectionModelTableEntry
from ..utilities import units
from ..utilities.config import CONST as UNIV_CONST
from ..utilities.miscmath import floattoint
from ..utilities.img_file_io import get_raw_as_hwl, LayerOffset
from ..utilities.tableio import readtable, writetable
from ..utilities.version import astropathversionregex
from .annotationxmlreader import AnnotationXMLReader
from .annotationpolygonxmlreader import ThingWithAnnotationInfos, XMLPolygonAnnotationReader, XMLPolygonAnnotationReaderWithOutline
from .argumentparser import ArgumentParserMoreRoots, DbloadArgumentParser, DeepZoomArgumentParser, GeomFolderArgumentParser, Im3ArgumentParser, ImageCorrectionArgumentParser, MaskArgumentParser, ParallelArgumentParser, SelectRectanglesArgumentParser, TempDirArgumentParser, XMLPolygonFileArgumentParser, ZoomFolderArgumentParser
from .csvclasses import AnnotationInfo, constantsdict, ExposureTime, MakeClinicalInfo, MergeConfig, RectangleFile
from .logging import getlogger, ThingWithLogger
from .rectangle import Rectangle, RectangleCollection, rectangleoroverlapfilter, RectangleReadComponentTiffSingleLayer, RectangleReadComponentTiffMultiLayer, RectangleReadIm3, RectangleReadIm3MultiLayer, RectangleCorrectedIm3SingleLayer, RectangleCorrectedIm3MultiLayer
from .overlap import Overlap, OverlapCollection, RectangleOverlapCollection
from .samplemetadata import SampleDef
from .workflowdependency import WorkflowDependencySlideID

class SampleBase(units.ThingWithPscale, ArgumentParserMoreRoots, ThingWithLogger, contextlib.ExitStack):
  """
  Base class for all sample classes.

  root: the Clinical_Specimen_i folder, which contains a bunch of SlideID folders
  samp: the SampleDef for this sample, or the SlideID for this sample as a string
  xmlfolders: possible places to look for xml metadata
              (will look by default in root/SlideID/im3/xml as well as
              with image files if they exist)

  uselogfiles, logthreshold, reraiseexceptions, logroot, mainlog, samplelog:
    these arguments get passed to getlogger
    logroot, by default, is the same as root
  """
  def __init__(self, root, samp, *, xmlfolders=None, uselogfiles=False, logthreshold=logging.NOTSET-100, reraiseexceptions=True, logroot=None, mainlog=None, samplelog=None, im3root=None, informdataroot=None, moremainlogroots=[], skipstartfinish=False, printthreshold=logging.DEBUG, Project=None, **kwargs):
    self.__root = pathlib.Path(root)
    self.samp = SampleDef(root=root, samp=samp, Project=Project)
    if not (self.root/self.SlideID).exists():
      raise FileNotFoundError(f"{self.root/self.SlideID} does not exist")
    if logroot is None: logroot = root
    self.__logroot = pathlib.Path(logroot)
    if im3root is None: im3root = root
    self.__im3root = pathlib.Path(im3root)
    if informdataroot is None: informdataroot = root
    self.__informdataroot = pathlib.Path(informdataroot)
    self.__logger = getlogger(module=self.logmodule(), root=self.logroot, samp=self.samp, uselogfiles=uselogfiles, threshold=logthreshold, reraiseexceptions=reraiseexceptions, mainlog=mainlog, samplelog=samplelog, moremainlogroots=moremainlogroots, skipstartfinish=skipstartfinish, printthreshold=printthreshold)
    self.__printlogger = getlogger(module=self.logmodule(), root=self.logroot, samp=self.samp, uselogfiles=False, threshold=logthreshold, skipstartfinish=skipstartfinish, printthreshold=printthreshold)
    if xmlfolders is None: xmlfolders = []
    self.__xmlfolders = xmlfolders
    self.__nentered = 0
    super().__init__(**kwargs)

    if not self.scanfolder.exists():
      raise OSError(f"{self.scanfolder} does not exist")
    if not self.scanfolder.is_dir():
      raise OSError(f"{self.scanfolder} is not a directory")

  @property
  def root(self): return self.__root
  @property
  def logroot(self): return self.__logroot
  @property
  def im3root(self): return self.__im3root
  @property
  def informdataroot(self): return self.__informdataroot
  @property
  def logger(self): return self.__logger
  @property
  def printlogger(self): return self.__printlogger
  @classmethod
  def usegloballogger(cls): return False

  @property
  def rootnames(self):
    return {"root", "logroot", "im3root", "informdataroot", *super().rootnames}

  @property
  def workflowkwargs(self):
    return {
      **super().workflowkwargs,
      "Scan": self.Scan,
    }

  @property
  def SampleID(self): return self.samp.SampleID
  @property
  def SlideID(self): return self.samp.SlideID
  @property
  def Project(self): return self.samp.Project
  @property
  def Cohort(self): return self.samp.Cohort
  @property
  def Scan(self): return self.samp.Scan
  @property
  def BatchID(self): return self.samp.BatchID
  @property
  def isGood(self): return self.samp.isGood
  def __bool__(self): return bool(self.samp)

  @property
  def mainfolder(self):
    """
    The folder where this sample's data lives
    """
    return self.root/self.SlideID

  @property
  def im3folder(self):
    """
    The sample's im3 folder
    """
    return self.im3root/self.SlideID/UNIV_CONST.IM3_DIR_NAME

  @property
  def scanfolder(self):
    """
    The sample's scan folder
    """
    return self.im3folder/f"Scan{self.Scan}"

  @property
  def qptifffilename(self):
    """
    The sample's qptiff image
    """
    return self.scanfolder/(self.SlideID+"_"+self.scanfolder.name+".qptiff")

  @property
  def componenttiffsfolder(self):
    """
    The sample's component tiffs folder
    """
    return self.informdataroot/self.SlideID/"inform_data"/"Component_Tiffs"

  def __getimageinfofromcomponenttiff(self):
    """
    Find the pscale and image dimensions from the component tiff.
    """
    try:
      componenttifffilename = next(self.componenttiffsfolder.glob(self.SlideID+"*_component_data.tif"))
    except StopIteration:
      raise FileNotFoundError(f"No component tiffs for {self}")
    with tifffile.TiffFile(componenttifffilename) as f:
      nlayers = None
      page = f.pages[0]
      resolutionunit = page.tags["ResolutionUnit"].value
      xresolution = page.tags["XResolution"].value
      xresolution = fractions.Fraction(*xresolution)
      yresolution = page.tags["YResolution"].value
      yresolution = fractions.Fraction(*yresolution)
      if xresolution != yresolution: raise ValueError(f"x and y have different resolutions {xresolution} {yresolution}")
      resolution = float(xresolution)
      kw = {
        tifffile.TIFF.RESUNIT.CENTIMETER: "centimeters",
      }[resolutionunit]
      pscale = float(units.Distance(pixels=resolution, pscale=1) / units.Distance(**{kw: 1}, pscale=1))
      height, width = units.distances(pixels=page.shape, pscale=pscale, power=1)

      return pscale, width, height, nlayers

  @property
  def possiblexmlfolders(self):
    """
    Possible places to look for xml metadata
    """
    return self.__xmlfolders + [folder/self.SlideID for folder in self.__xmlfolders] + [self.im3folder/"xml"]
  @property
  def xmlfolder(self):
    """
    The folder where the xml metadata lives.
    It's the first one in possiblexmlfolders that contains SlideID.Parameters.xml
    """
    possibilities = self.possiblexmlfolders
    for possibility in possibilities:
      if (possibility/(self.SlideID+".Parameters.xml")).exists(): return possibility
    raise FileNotFoundError(f"Didn't find {self.SlideID}.Parameters.xml in any of these folders:\n" + ", ".join(str(_) for _ in possibilities))

  @property
  def parametersxmlfile(self):
    return self.xmlfolder/(self.SlideID+".Parameters.xml")
  @property
  def fullxmlfile(self):
    return self.xmlfolder/(self.SlideID+".Full.xml")
  @property
  def annotationsxmlfile(self):
    return self.scanfolder/(self.SlideID+"_"+self.scanfolder.name+"_annotations.xml")

  def __getimageinfofromXMLfiles(self):
    """
    Find the pscale, image dimensions, and number of layers from the XML metadata.
    """
    with open(self.parametersxmlfile, "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="/IM3Fragment/D"):
        if node.xml_attrs["name"] == "Shape":
          width, height, nlayers = (int(_) for _ in str(node).split())
        if node.xml_attrs["name"] == "MillimetersPerPixel":
          pscale = 1e-3/float(node)

    try:
      width *= units.onepixel(pscale=pscale)
      height *= units.onepixel(pscale=pscale)
    except NameError:
      raise IOError(f'Couldn\'t find Shape and/or MillimetersPerPixel in {self.parametersxmlfile}')

    return pscale, width, height, nlayers

  @methodtools.lru_cache()
  @property
  def samplelocation(self):
    """
    Find the location of the image on the slide from the xml metadata
    """
    with open(self.parametersxmlfile, "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="/IM3Fragment/D"):
        if node.xml_attrs["name"] == "SampleLocation":
          return np.array([float(_) for _ in str(node).split()]) * units.onemicron(pscale=self.apscale)

  @methodtools.lru_cache()
  def __getcamerastateparameter(self, parametername):
    """
    Find info from the CameraState Parameters block in the xml metadata
    """
    with open(self.fullxmlfile, "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="/IM3/G/G/G/G/G/G/G/G"):
        if node.xml_attrs["name"] == "CameraState":
          for G in node["G"]["G"]:
            if G.xml_attrs["name"] == "Parameters":
              for subG in G["G"]["G"]["G"]:
                name = value = None
                for D in subG["D"]:
                  if D.xml_attrs["name"] == "Name":
                    name = str(D)
                  if D.xml_attrs["name"] == "Value":
                    value = int(str(D))
                assert name is not None is not value
                if name == parametername: return value
      assert False

  @property
  def resolutionbits(self):
    """
    Find the number of resolution bits from the xml metadata
    """
    return self.__getcamerastateparameter("ResolutionBits")

  @property
  def gainfactor(self):
    """
    Find the camera gain factor from the xml metadata
    """
    return self.__getcamerastateparameter("GainFactor")

  @methodtools.lru_cache()
  def __getcamerabinning(self, xory):
    """
    Find the camera binning from the xml metadata
    """
    with open(self.fullxmlfile, "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="/IM3/G/G/G/G/G/G/G/G"):
        if node.xml_attrs["name"] == "CameraState":
          for G in node["G"]["G"]:
            if G.xml_attrs["name"] == "Binning":
              for D in G["G"]["D"]:
                if D.xml_attrs["name"] == xory:
                  return int(str(D)) * self.onepixel
    assert False

  @property
  def camerabinningx(self):
    """
    Find the camera binning in x
    """
    return self.__getcamerabinning("X")
  @property
  def camerabinningy(self):
    """
    Find the camera binning in y
    """
    return self.__getcamerabinning("Y")

  @methodtools.lru_cache()
  @property
  def nlayersim3(self):
    """
    Find the number of im3 layers from the xml metadata
    """
    with open(self.parametersxmlfile, "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="/IM3Fragment/D"):
        if node.xml_attrs["name"] == "Shape":
          width, height, nlayers = (int(_) for _ in str(node).split())
          return nlayers
      else:
        raise IOError(f'Couldn\'t find Shape in {self.parametersxmlfile}')

  @classmethod
  def getnlayersunmixed(cls, componenttiffsfolder):
    filenames = [componenttiffsfolder/"batch_procedure.ifp", componenttiffsfolder/"batch_procedure.ifr"]
    try:
      filename = next(_ for _ in filenames if _.exists())
    except StopIteration:
      raise FileNotFoundError("Didn't find any batch procedure files: " + ", ".join(str(_) for _ in filenames))
    with open(filename, "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="AllComponents"):
        return int(node.xml_attrs["dim"])

  @methodtools.lru_cache()
  @property
  def nlayersunmixed(self):
    """
    Find the number of component tiff layers from the xml metadata
    """
    return self.getnlayersunmixed(self.componenttiffsfolder)

  def _getimageinfos(self):
    """
    Try to find the image info from a bunch of sources.
    Called by getimageinfo.
    """
    result = {}
    try:
      result["component tiff"] = self.__getimageinfofromcomponenttiff()
    except FileNotFoundError:
      result["component tiff"] = None
    try:
      result["xml files"] = self.__getimageinfofromXMLfiles()
    except (FileNotFoundError, IOError):
      result["xml files"] = None
    return result

  @methodtools.lru_cache()
  def getimageinfo(self):
    """
    Try to find the image info from a bunch of sources.
    Gives a warning or error if they are inconsistent, then returns one of them.
    """
    results = self._getimageinfos()
    result = None
    warnfunction = None
    for k, v in sorted(results.items(), key=lambda kv: kv[1] is None or any(_ is None for _ in kv[1])):
      if v is None: continue
      pscale, width, height, layers = v
      if result is None:
        result = resultpscale, resultwidth, resultheight, resultlayers = v
        resultk = k
      elif None is not layers != resultlayers is not None:
        raise ValueError(f"Found different numbers of layers from different sources: {resultk}: {resultlayers}, {k}: {layers}")
      elif np.allclose(units.microns(result[:3], power=[0, 1, 1], pscale=result[0]), units.microns(v[:3], power=[0, 1, 1], pscale=v[0]), rtol=1e-6):
        continue
      elif np.allclose(units.microns(result[:3], power=[0, 1, 1], pscale=result[0]), units.microns(v[:3], power=[0, 1, 1], pscale=v[0]), rtol=1e-6):
        if warnfunction != self.logger.warningglobalonenter: warnfunction = self.logger.warningonenter
      else:
        warnfunction = self.logger.warningglobalonenter

    if warnfunction is not None:
      fmt = "{:30} {:30} {:30} {:30}"
      warninglines = [
        "Found inconsistent image infos from different sources:",
        fmt.format("source", "pscale", "field width", "field height"),
      ]
      for k, v in results.items():
        if v is None: continue
        warninglines.append(fmt.format(k, *(str(_) for _ in v)))
      for warningline in warninglines:
        warnfunction(warningline)

    if result is None:
      raise FileNotFoundError("Didn't find any of the possible ways of finding image info: "+", ".join(results))

    return result

  @property
  def mergeconfigcsv(self):
    return self.root/"Batch"/f"MergeConfig_{self.BatchID:02d}.csv"
  @property
  def mergeconfig(self):
    return self.readtable(self.mergeconfigcsv, MergeConfig)

  @property
  def samplelog(self):
    """
    The sample log file, which contains detailed logging info
    """
    return self.logger.samplelog
  @property
  def mainlogs(self):
    """
    The cohort log file, which contains basic logging info
    """
    return self.logger.mainlogs

  @property
  def pscale(self):
    """
    Pixels per micron of the im3 image.
    """
    return self.getimageinfo()[0]
  @property
  def fwidth(self):
    """
    Width of the im3 image.
    """
    return self.getimageinfo()[1]
  @property
  def fheight(self):
    """
    Height of the im3 image.
    """
    return self.getimageinfo()[2]
  @property
  def flayers(self):
    """
    Number of layers in the im3 image.
    """
    layers = self.getimageinfo()[3]
    if layers is None: raise FileNotFoundError("Couldn't get image info from any source that has flayers")
    return layers

  @methodtools.lru_cache()
  def getXMLplan(self):
    """
    Read the annotations xml file to get the structure of the
    image as well as the microscope name (really the name of the
    computer that processed the image).
    """
    xmlfile = self.annotationsxmlfile
    try:
      xmlfolder = self.xmlfolder
    except FileNotFoundError:
      xmlfolder = None
    reader = AnnotationXMLReader(xmlfile, xmlfolder=xmlfolder, pscale=self.pscale)
    return reader.rectangles, reader.globals, reader.perimeters, reader.microscopename

  @property
  def microscopename(self):
    """
    Name of the computer that processed the image.
    """
    return self.getXMLplan()[3]

  def __enter__(self):
    self.__nentered += 1
    self.enter_context(self.logger)
    return super().__enter__()

  def __exit__(self, *exc):
    self.__nentered -= 1
    return super().__exit__(*exc)

  def enter_context(self, *args, **kwargs):
    if not self.__nentered:
      raise ValueError(f"Have to use {self} in a with statement if you want to enter_context")
    return super().enter_context(*args, **kwargs)

  @classmethod
  def logstartregex(cls): return rf"(?:START: )?{cls.logmodule()} {astropathversionregex.pattern}$"
  @classmethod
  def logendregex(cls): return rf"end {cls.logmodule()}$|FINISH: {cls.logmodule()} v[0-9a-f.devgd+]+$"

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("SlideID", help="The SlideID of the sample to run")
    p.add_argument("--project", type=int, help="Project number (used to identify the AstropathAPIDdef file)")
    return p

  @classmethod
  def defaultunits(cls):
    return "fast_pixels"

  @classmethod
  def runfromargsdicts(cls, *, initkwargs, runkwargs, misckwargs):
    """
    Run the sample from command line arguments.
    """
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
      "samp": parsed_args_dict.pop("SlideID"),
      "Project": parsed_args_dict.pop("project"),
    }

  @abc.abstractmethod
  def run(self, **kwargs):
    "actually run whatever is supposed to be run on the sample"

  @property
  def clinicalfolder(self): return self.root/"Clinical"
  @methodtools.lru_cache()
  @property
  def clinicalinfo(self, *, filenamepattern="*.csv"):
    results = []
    for filename in self.clinicalfolder.glob(filenamepattern):
      ClinicalInfo = MakeClinicalInfo(filename)
      clinicalinfos = readtable(filename, ClinicalInfo)
      for c in clinicalinfos:
        if c.SlideID == self.SlideID:
          results.append(c)
    try:
      result, = results
    except ValueError:
      if not results:
        raise ValueError(f"Didn't find clinical info for {self.SlideID}")
      raise ValueError("Found multiple clinical infos:\n"+"\n".join(str(_) for _ in results))
    return result
  @property
  def REDCapID(self):
    return self.clinicalinfo.REDCapID

class WorkflowSample(SampleBase, WorkflowDependencySlideID):
  """
  Base class for a sample that will be used in a workflow,
  i.e. it takes in input files and creates output files.
  It contains functions to assess the status of the run.
  """

  @abc.abstractmethod
  def inputfiles(self, **kwargs):
    """
    Required files that have to be present for this step to run
    """
    return []

  @classmethod
  @abc.abstractmethod
  def workflowdependencyclasses(cls, **kwargs):
    """
    Previous steps that this step depends on
    """
    return []

  def workflowdependencies(self, **kwargs):
    return [(dependencycls, self.SlideID) for dependencycls in self.workflowdependencyclasses(**kwargs)]

  def joblock(self, corruptfiletimeout=datetime.timedelta(minutes=10), **kwargs):
    return job_lock.JobLock(self.samplelog.with_suffix(".lock"), corruptfiletimeout=corruptfiletimeout, mkdir=True, **kwargs)

class DbloadSampleBase(SampleBase, DbloadArgumentParser):
  """
  Base class for any sample that uses the csvs in the dbload folder.
  If the dbload folder is initialized already, you should instead inherit
  from DbloadSample to get more functionality.

  dbloadroot: A different root to use to find the dbload (default: same as root)
  """
  def __init__(self, *args, dbloadroot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if dbloadroot is None:
      dbloadroot = self.mainfolder.parent
    else:
      dbloadroot = pathlib.Path(dbloadroot)
    self.__dbloadroot = dbloadroot
  @property
  def dbload(self):
    """
    The folder where the csv files live.
    """
    return self.__dbloadroot/self.SlideID/UNIV_CONST.DBLOAD_DIR_NAME
  @property
  def dbloadroot(self): return self.__dbloadroot

  @property
  def rootnames(self):
    return {"dbloadroot", *super().rootnames}

  def csv(self, csv):
    """
    Full path to the csv file identified by csv.
    """
    return self.dbload/f"{self.SlideID}_{csv}.csv"
  def readcsv(self, csv, *args, **kwargs):
    """
    Read the csv file using readtable.
    """
    return self.readtable(self.csv(csv), *args, **kwargs)
  def writecsv(self, csv, *args, **kwargs):
    """
    Write the csv file using writetable.
    """
    return writetable(self.csv(csv), *args, logger=self.logger, **kwargs)

  @methodtools.lru_cache()
  def qptiffjpgimage(self):
    """
    Read the qptiff jpg thumbnail as an image.
    """
    return cv2.imread(os.fspath(self.dbload/(self.SlideID+UNIV_CONST.QPTIFF_SUFFIX)))

class DbloadSample(DbloadSampleBase, units.ThingWithQpscale, units.ThingWithApscale):
  """
  Base class for any sample that uses the csvs in the dbload folder
  after the folder has been set up.

  dbloadroot: A different root to use to find the dbload (default: same as root)
  """
  def __getimageinfofromconstants(self):
    """
    Find the image size and pscale from the constants csv.
    """
    dct = constantsdict(self.csv("constants"))

    fwidth    = dct["fwidth"]
    fheight   = dct["fheight"]
    flayers   = dct["flayers"]
    pscale    = float(dct["pscale"])

    return pscale, fwidth, fheight, flayers

  def _getimageinfos(self):
    result = super()._getimageinfos()

    try:
      result["constants.csv"] = self.__getimageinfofromconstants()
    except (FileNotFoundError, KeyError):
      result["constants.csv"] = None

    return result

  @methodtools.lru_cache()
  @property
  def constantsdict(self):
    """
    Read the constants.csv file into a dict.
    """
    return constantsdict(self.csv("constants"), pscale=self.pscale)

  @property
  def position(self):
    """
    x and y position of the tissue from constants.csv
    """
    return np.array([self.constantsdict["xposition"], self.constantsdict["yposition"]])
  @property
  def nclip(self):
    """
    number of pixels to clip off the edge of the image for calibration
    """
    return self.constantsdict["nclip"]
  @property
  def qpscale(self):
    """
    Pixels/micron of the qptiff layer used for the jpg thumbnail
    """
    return self.constantsdict["qpscale"]
  @property
  def apscale(self):
    """
    Pixels/micron of the first qptiff layer
    """
    return self.constantsdict["apscale"]
  @property
  def margin(self):
    """
    Margin to add outside the image area in the wsi
    Default 0 for backwards compatibility
    """
    return self.constantsdict.get("margin", 0)

class MaskSampleBase(SampleBase, MaskArgumentParser):
  """
  Base class for any sample that uses the masks in im3/meanimage

  maskroot: A different root to use to find the masks (default: same as root)
  """
  def __init__(self, *args, maskroot=None, maskfilesuffix=None, **kwargs):
    super().__init__(*args, **kwargs)
    if maskroot is None: maskroot = self.im3root
    self.__maskroot = pathlib.Path(maskroot)
    self.__maskfolder = None
    if maskfilesuffix is None: maskfilesuffix = self.defaultmaskfilesuffix
    self.__maskfilesuffix = maskfilesuffix
  @property
  def maskroot(self): return self.__maskroot
  @property
  def maskfilesuffix(self): return self.__maskfilesuffix

  @property
  def rootnames(self): return {"maskroot", *super().rootnames}

  @property
  def maskfolder(self):
    if self.__maskfolder is not None :
      return self.__maskfolder
    else :
      result = self.im3folder/UNIV_CONST.MEANIMAGE_DIRNAME/FF_CONST.IMAGE_MASKING_SUBDIR_NAME
      if self.maskroot != self.im3root:
        result = self.maskroot/result.relative_to(self.im3root)
      return result
  @maskfolder.setter
  def maskfolder(self,mf):
    if self.__maskfolder is None :
      self.__maskfolder=mf
    else :
      raise ValueError(f'ERROR: maskfolder has already been set to {self.__maskfolder}!')

class MaskWorkflowSampleBase(MaskSampleBase, WorkflowSample):
  @property
  def workflowkwargs(self):
    return {
      **super().workflowkwargs,
      "maskfilesuffix": self.maskfilesuffix,
    }

class Im3SampleBase(SampleBase, Im3ArgumentParser):
  """
  Base class for any sample that uses sharded im3 images.
  shardedim3root: Root location of the im3 images.
         (The images are in shardedim3root/SlideID)
  """
  def __init__(self, root, shardedim3root, samp, *args, **kwargs):
    super().__init__(root=root, samp=samp, *args, **kwargs)
    self.shardedim3root = pathlib.Path(shardedim3root)

  @property
  def root1(self): return self.root

  @property
  def rootnames(self): return {"shardedim3root", *super().rootnames}

  @property
  def possiblexmlfolders(self):
    return super().possiblexmlfolders + [self.shardedim3root/self.SlideID]

class ZoomFolderSampleBase(SampleBase, ZoomFolderArgumentParser):
  """
  Base class for any sample that uses the zoomed "big" or "wsi" images.
  zoomroot: Root location of the zoomed images.
            (The images are in zoomroot/SlideID/big and zoomroot/SlideID/wsi)
  """
  def __init__(self, *args, zoomroot, **kwargs):
    super().__init__(*args, **kwargs)
    self.__zoomroot = pathlib.Path(zoomroot)
  @property
  def rootnames(self): return {"zoomroot", *super().rootnames}
  @property
  def zoomroot(self): return self.__zoomroot
  @property
  def bigfolder(self): return self.zoomroot/self.SlideID/"big"
  @property
  def wsifolder(self): return self.zoomroot/self.SlideID/"wsi"

  zmax = 9
  ztiff = 8

  def bigfilename(self, layer, tilex, tiley):
    """
    Zoom filename for a given layer and tile.
    """
    return self.bigfolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-X{tilex}-Y{tiley}-big.png"
  def wsifilename(self, layer):
    """
    Wsi filename for a given layer.
    """
    return self.wsifolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-wsi.png"
  def wsitifffilename(self, layers):
    name = f"{self.SlideID}-Z{self.ztiff}"
    if layers == "color":
      name += "-color"
    elif frozenset(layers) != frozenset(range(1, self.nlayersunmixed+1)):
      name += "-L" + "".join(str(l) for l in sorted(layers))
    name += "-wsi.tiff"
    return self.wsifolder/name

class DeepZoomSampleBase(SampleBase, DeepZoomArgumentParser):
  """
  Base class for any sample that uses the deepzoomed images.
  deepzoomroot: Root location of the deepzoomed images.
                (The images are in deepzoomroot/SlideID)
  """
  def __init__(self, *args, deepzoomroot, **kwargs):
    super().__init__(*args, **kwargs)
    self.__deepzoomroot = pathlib.Path(deepzoomroot)
  @property
  def rootnames(self): return {"deepzoomroot", *super().rootnames}
  @property
  def deepzoomroot(self): return self.__deepzoomroot
  @property
  def deepzoomfolder(self): return self.deepzoomroot/self.SlideID

class GeomSampleBase(SampleBase, GeomFolderArgumentParser):
  """
  Base class for any sample that uses the _cellgeomload.csv files

  geomroot: A different root to use to find the cellgeomload (default: same as root)
  """
  def __init__(self, *args, geomroot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if geomroot is None: geomroot = self.root
    self.__geomroot = pathlib.Path(geomroot)

  @property
  def rootnames(self): return {"geomroot", *super().rootnames}
  @property
  def geomroot(self): return self.__geomroot
  @property
  def geomfolder(self):
    return self.geomroot/self.SlideID/"geom"

class CellPhenotypeSampleBase(SampleBase):
  """
  Base class for any sample that uses the _cleaned_phenotype_table.csv files

  phenotyperoot: A different root to use to find the Phenotyped folder (default: same as root)
  """
  def __init__(self, *args, phenotyperoot=None, **kwargs):
    super().__init__(*args, **kwargs)
    if phenotyperoot is None: phenotyperoot = self.informdataroot
    self.__phenotyperoot = pathlib.Path(phenotyperoot)

  @property
  def rootnames(self): return {"phenotyperoot", *super().rootnames}
  @property
  def phenotyperoot(self): return self.__phenotyperoot
  @property
  def phenotypefolder(self):
    return self.phenotyperoot/self.SlideID/"inform_data"/"Phenotyped"
  @property
  def phenotypetablesfolder(self):
    return self.phenotypefolder/"Results"/"Tables"
  @property
  def phenotypeQAQCtablesfolder(self):
    return self.phenotypefolder/"Results"/"QA_QC"/"Tables_QA_QC"

class SelectLayersSample(SampleBase):
  """
  Base class for any sample that needs a layer selection.
  """
  def __init__(self, *args, layer=None, layers=None, **kwargs):
    if layer != "setlater" != layers:
      self.setlayers(layer=layer, layers=layers)
    super().__init__(*args, **kwargs)

  def setlayers(self, layer=None, layers=None):
    try:
      self.__layers
    except AttributeError:
      pass
    else:
      raise AttributeError("Already called setlayers for this sample")
    if self.multilayer:
      if layer is not None:
        raise TypeError(f"Can't provide layer for a multilayer sample {type(self).__name__}")
    else:
      if layers is not None:
        raise TypeError(f"Can't provide layers for a single layer sample {type(self).__name__}")
      if layer is None:
        layer = 1
      self.__layer = layer
      layers = layer,

    self.__layers = layers

  @property
  def layers(self):
    result = self.__layers
    if result is None: return range(1, self.nlayers+1)
    return result

  @property
  def layer(self):
    if self.multilayer:
      raise TypeError(f"Can't get layer for a multilayer sample {type(self).__name__}")
    return self.__layer

  @abc.abstractproperty
  def nlayers(self): pass

  multilayer = False #can override in subclasses

class SelectLayersIm3(SelectLayersSample):
  @property
  def nlayers(self): return self.nlayersim3

class SelectLayersComponentTiff(SelectLayersSample):
  @property
  def nlayers(self): return self.nlayersunmixed

class ReadRectanglesBase(RectangleCollection, SampleBase, SelectRectanglesArgumentParser):
  """
  Base class for any sample that reads HPF info from any source.
  selectrectangles: filter for selecting rectangles (a list of ids or a function)
  """
  def __init__(self, *args, selectrectangles=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.__rectanglefilter = rectangleoroverlapfilter(selectrectangles)
    self.__initedrectangles = False

  def initrectangles(self):
    self.__initedrectangles = True
    self.__rectangles  = self.readallrectangles()
    self.__rectangles = [r for r in self.rectangles if self.__rectanglefilter(r)]

  @abc.abstractmethod
  def readallrectangles(self):
    """
    The function that actually reads the HPF info and returns a list of rectangletype
    """
  rectangletype = Rectangle #can be overridden in subclasses
  @property
  def rectangleextrakwargs(self):
    """
    Extra keyword arguments to give to the rectangle constructor
    besides the ones determined from the direct source in readallrectangles
    """
    kwargs = {
      "pscale": self.pscale,
    }
    try:
      kwargs.update({
        "xmlfolder": self.xmlfolder,
      })
    except FileNotFoundError:
      pass
    return kwargs

  @property
  def _rectangles(self):
    assert False
  @_rectangles.setter
  def _rectangles(self, value):
    self.__rectangles = value

  @property
  def rectangles(self):
    if not self.__initedrectangles: self.initrectangles()
    return self.__rectangles

class ReadRectanglesWithLayers(ReadRectanglesBase, SelectLayersSample):
  """
  Base class for any sample that reads rectangles
  and needs a layer selection.
  """
  @property
  def rectangleextrakwargs(self):
    kwargs = {
      **super().rectangleextrakwargs,
      "nlayers": self.nlayers,
    }
    if self.multilayer:
      kwargs.update({
        "layers": self.layers,
      })
    else:
      kwargs.update({
        "layer": self.layer,
      })
    return kwargs

class ReadRectanglesIm3Base(ReadRectanglesWithLayers, Im3SampleBase, SelectLayersIm3):
  """
  Base class for any sample that loads images from an im3 file.
  filetype: "raw", "flatWarp", or "camWarp"
  layer or layers: the layer or layers to read, depending on whether
                   the class uses multilayer images or not
  readlayerfile: whether or not to read from a file with a single layer, e.g. .fw01
  """

  def __init__(self, *args, filetype, readlayerfile=None, **kwargs):
    self.__filetype = filetype

    if readlayerfile is None: readlayerfile = not self.multilayer

    if self.multilayer and readlayerfile:
      raise ValueError(f"Can't read a layer file for a multilayer sample {type(self).__name__}")

    self.__readlayerfile = readlayerfile

    super().__init__(*args, **kwargs)

  @property
  def nlayers(self):
    if self.__readlayerfile: return 1
    return super().nlayers
  @property
  def rectangletype(self):
    if self.multilayer:
      return RectangleReadIm3MultiLayer
    else:
      return RectangleReadIm3
  @property
  def rectangleextrakwargs(self):
    kwargs = {
      **super().rectangleextrakwargs,
      "im3folder": self.shardedim3root/self.SlideID,
      "im3filetype": self.filetype,
      "width": self.fwidth,
      "height": self.fheight,
    }
    if self.multilayer:
      pass
    else:
      kwargs.update({
        "readlayerfile": self.__readlayerfile,
      })
    return kwargs

  @property
  def filetype(self): return self.__filetype

class ReadRectanglesComponentTiffBase(ReadRectanglesWithLayers, SelectLayersComponentTiff):
  """
  Base class for any sample that loads images from a component tiff file.
  layer or layers: the layer or layers to read, depending on whether
                   the class uses multilayer images or not
  """
  @property
  def with_seg(self): return False

  @property
  def rectangletype(self):
    if self.multilayer:
      return RectangleReadComponentTiffMultiLayer
    else:
      return RectangleReadComponentTiffSingleLayer
  @property
  def rectangleextrakwargs(self):
    kwargs = {
      **super().rectangleextrakwargs,
      "imagefolder": self.componenttiffsfolder,
      "with_seg": self.with_seg,
    }
    return kwargs

class ReadRectanglesOverlapsBase(ReadRectanglesBase, RectangleOverlapCollection, OverlapCollection, SampleBase):
  """
  Base class for any sample that reads rectangles and overlaps from any source.
  selectoverlaps: filter for which overlaps to use (a list of overlap ids or a function that returns True or False).
  onlyrectanglesinoverlaps: an additional selection that only includes rectangles that are in selected overlaps.
  """

  def __init__(self, *args, selectoverlaps=None, onlyrectanglesinoverlaps=False, **kwargs):
    super().__init__(*args, **kwargs)

    _overlapfilter = rectangleoroverlapfilter(selectoverlaps)
    self.__overlapfilter = lambda o: _overlapfilter(o) and o.p1 in self.rectangleindices and o.p2 in self.rectangleindices
    self.__onlyrectanglesinoverlaps = onlyrectanglesinoverlaps

  def initrectangles(self):
    super().initrectangles()
    self.__overlaps  = self.readalloverlaps()
    self.__overlaps = [o for o in self.overlaps if self.__overlapfilter(o)]
    if self.__onlyrectanglesinoverlaps:
      self._rectangles = [r for r in self.rectangles if self.selectoverlaprectangles(r)]

  @abc.abstractmethod
  def readalloverlaps(self): pass
  @property
  def overlapextrakwargs(self):
    return {"pscale": self.pscale, "rectangles": self.rectangles, "nclip": self.nclip}

  @property
  def overlaps(self):
    self.rectangles #make sure initrectangles() has been called
    return self.__overlaps

class ReadRectanglesOverlapsIm3Base(ReadRectanglesOverlapsBase, ReadRectanglesIm3Base):
  """
  Base class for any sample that reads rectangles and overlaps from any source
  and loads the rectangle images from im3 files.
  """

class ReadRectanglesOverlapsComponentTiffBase(ReadRectanglesOverlapsBase, ReadRectanglesComponentTiffBase):
  """
  Base class for any sample that reads rectangles and overlaps from any source
  and loads the rectangle images from component tiff files.
  """

class ReadRectanglesDbload(ReadRectanglesBase, DbloadSample, DbloadArgumentParser, SelectRectanglesArgumentParser):
  """
  Base class for any sample that reads rectangles from the dbload folder.
  """
  @property
  def rectangleextrakwargs(self):
    result = {
      **super().rectangleextrakwargs,
    }
    try:
      result.update({
        "allexposures": self.readcsv("exposures", ExposureTime)
      })
    except FileNotFoundError:
      pass
    return result

  @property
  def rectanglecsv(self): return "rect"
  def readallrectangles(self, **extrakwargs):
    return self.readcsv(self.rectanglecsv, self.rectangletype, extrakwargs={**self.rectangleextrakwargs, **extrakwargs})

class ReadRectanglesDbloadIm3(ReadRectanglesIm3Base, ReadRectanglesDbload):
  """
  Base class for any sample that reads rectangles from the dbload folder
  and loads the rectangle images from im3 files.
  """

class ReadRectanglesDbloadComponentTiff(ReadRectanglesComponentTiffBase, ReadRectanglesDbload):
  """
  Base class for any sample that reads rectangles from the dbload folder
  and loads the rectangle images from component tiff files.
  """

class ReadRectanglesOverlapsDbload(ReadRectanglesOverlapsBase, ReadRectanglesDbload):
  """
  Base class for any sample that reads rectangles and overlaps from the dbload folder.
  """
  @property
  def overlapcsv(self): return "overlap"
  def readalloverlaps(self, *, overlaptype=None, **extrakwargs):
    if overlaptype is None: overlaptype = self.overlaptype
    return self.readcsv(self.overlapcsv, overlaptype, filter=lambda row: row["p1"] in self.rectangleindices and row["p2"] in self.rectangleindices, extrakwargs={**self.overlapextrakwargs, **extrakwargs})

class ReadRectanglesOverlapsDbloadIm3(ReadRectanglesOverlapsIm3Base, ReadRectanglesOverlapsDbload, ReadRectanglesDbloadIm3):
  """
  Base class for any sample that reads rectangles and overlaps from the dbload folder
  and loads the rectangle images from im3 files.
  """

class ReadRectanglesOverlapsDbloadComponentTiff(ReadRectanglesOverlapsComponentTiffBase, ReadRectanglesOverlapsDbload, ReadRectanglesDbloadComponentTiff):
  """
  Base class for any sample that reads rectangles and overlaps from the dbload folder
  and loads the rectangle images from component tiff files.
  """

class XMLLayoutReader(SampleBase):
  """
  Base class for any sample that reads the HPF layout from the XML metadata.
  """
  def __init__(self, *args, checkim3s=False, **kwargs):
    self.__checkim3s = checkim3s
    super().__init__(*args, **kwargs)

  @methodtools.lru_cache()
  def getrectanglelayout(self):
    """
    Find the rectangle layout from both the XML metadata
    and the im3 files, compare them, and return the rectangles.
    """
    rectangles, globals, perimeters, microscopename = self.getXMLplan()
    self.fixM2(rectangles)
    self.fixrectanglefilenames(rectangles)
    self.fixduplicaterectangles(rectangles)
    rectanglefiles = self.getdir()
    maxtimediff = datetime.timedelta(0)
    for r in rectangles[:]:
      rfs = {rf for rf in rectanglefiles if np.all(rf.cxvec == r.cxvec)}
      assert len(rfs) <= 1
      if not rfs:
        cx, cy = floattoint((r.cxvec / self.onemicron).astype(float))
        errormessage = f"File {self.SlideID}_[{cx},{cy}].im3 (expected from annotations) does not exist"
        if self.__checkim3s:
          raise FileNotFoundError(errormessage)
        else:
          self.logger.warningglobalonenter(errormessage)
        rectangles.remove(r)
      else:
        rf = rfs.pop()
        maxtimediff = max(maxtimediff, abs(rf.t-r.t))
    if maxtimediff >= datetime.timedelta(seconds=5):
      self.logger.warningonenter(f"Biggest time difference between annotation and file mtime is {maxtimediff}")
    rectangles.sort(key=lambda x: x.t)
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i
    if not rectangles:
      raise ValueError("No layout annotations")
    return rectangles

  def fixM2(self, rectangles):
    """
    Fix any _M2 in the rectangle filenames
    """
    for rectangle in rectangles[:]:
      if "_M2" in rectangle.file:
        duplicates = [r for r in rectangles if r is not rectangle and np.all(r.cxvec == rectangle.cxvec)]
        if not duplicates:
          rectangle.file = rectangle.file.replace("_M2", "")
        for d in duplicates:
          rectangles.remove(d)
        self.logger.warningglobalonenter(f"{rectangle.file} has _M2 in the name.  {len(duplicates)} other duplicate rectangles.")
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i

  def fixrectanglefilenames(self, rectangles):
    """
    Fix rectangle filenames if the coordinates are messed up
    """
    for r in rectangles:
      expected = self.SlideID+f"_[{floattoint(float(r.cx/r.onemicron)):d},{floattoint(float(r.cy/r.onemicron)):d}]{UNIV_CONST.IM3_EXT}"
      actual = r.file
      if expected != actual:
        self.logger.warningglobalonenter(f"rectangle at ({r.cx}, {r.cy}) has the wrong filename {actual}.  Changing it to {expected}.")
      r.file = expected

  def fixduplicaterectangles(self, rectangles):
    """
    Remove duplicate rectangles from the list
    """
    seen = set()
    for r in rectangles[:]:
      if tuple(r.cxvec) in seen: continue
      seen.add(tuple(r.cxvec))
      duplicates = [r2 for r2 in rectangles if r2 is not r and np.all(r2.cxvec == r.cxvec)]
      if not duplicates: continue
      for r2 in duplicates:
        if r2.file != r.file:
          raise ValueError(f"Multiple rectangles at {r.cxvec} with different filenames {', '.join(r3.file for r3 in [r]+duplicates)}")
      self.logger.warningglobalonenter(f"annotations.xml has the rectangle at {r.cxvec} with filename {r.file} {len(duplicates)+1} times")
      for r2 in [r]+duplicates[:-1]:
        rectangles.remove(r2)
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i

  @methodtools.lru_cache()
  def getdir(self):
    """
    List all rectangles that have im3 files.
    """
    folder = self.scanfolder/"MSI"
    if not folder.exists():
      folder = self.scanfolder/"flatw"
    im3s = folder.glob(f"*{UNIV_CONST.IM3_EXT}")
    result = []
    for im3 in im3s:
      regex = self.SlideID+r"_\[([0-9]+),([0-9]+)\]"+UNIV_CONST.IM3_EXT
      match = re.match(regex, im3.name)
      if not match:
        raise ValueError(f"Unknown im3 filename {im3}, should match {regex}")
      x = int(match.group(1)) * self.onemicron
      y = int(match.group(2)) * self.onemicron
      t = datetime.datetime.fromtimestamp(os.path.getmtime(im3)).astimezone()
      result.append(
        RectangleFile(
          cx=x,
          cy=y,
          t=t,
          pscale=self.pscale,
        )
      )
    return result

  @methodtools.lru_cache()
  def getoverlaps(self, *, overlaptype=Overlap):
    """
    Calculate all overlaps between rectangles.
    """
    overlaps = []
    for r1, r2 in itertools.product(self.rectangles, repeat=2):
      if r1 is r2: continue
      if np.all(abs(r1.cxvec - r2.cxvec) < r1.shape):
        tag = int(np.sign(r1.cx-r2.cx)) + 3*int(np.sign(r1.cy-r2.cy)) + 5
        overlaps.append(
          overlaptype(
            n=len(overlaps)+1,
            p1=r1.n,
            p2=r2.n,
            x1=r1.x,
            y1=r1.y,
            x2=r2.x,
            y2=r2.y,
            tag=tag,
            nclip=self.nclip,
            rectangles=(r1, r2),
            pscale=self.pscale,
            readingfromfile=False,
          )
        )
    return overlaps

class SampleWithAnnotationInfos(SampleBase, ThingWithAnnotationInfos):
  def readtable(self, filename, rowclass, *, extrakwargs=None, **kwargs):
    if extrakwargs is None: extrakwargs = {}
    if issubclass(rowclass, AnnotationInfo):
      extrakwargs["scanfolder"] = self.scanfolder
    return super().readtable(filename=filename, rowclass=rowclass, extrakwargs=extrakwargs, **kwargs)

class XMLPolygonAnnotationFileSample(SampleWithAnnotationInfos, XMLPolygonFileArgumentParser):
  """
  Base class for any sample that uses the XML annotations file.
  """
  def __init__(self, *args, annotationsxmlregex=None, **kwargs):
    if annotationsxmlregex is not None: annotationsxmlregex = re.compile(annotationsxmlregex)
    self.__annotationsxmlregex = annotationsxmlregex
    super().__init__(*args, **kwargs)

  @property
  def workflowkwargs(self):
    return {
      **super().workflowkwargs,
      "annotationsxmlregex": self.__annotationsxmlregex,
    }

  @methodtools.lru_cache()
  @property
  def annotationspolygonsxmlfile(self):
    candidates = {
      filename
      for filename in self.scanfolder.glob(f"*{self.SlideID}*annotations.polygons*.xml")
      if self.__annotationsxmlregex is None or self.__annotationsxmlregex.match(filename.name)
    }
    default = self.scanfolder/f"{self.SlideID}_{self.scanfolder.name}.annotations.polygons.xml"
    try:
      candidate, = candidates
    except ValueError:
      if candidates:
        if self.__annotationsxmlregex is None:
          raise IOError("Found multiple annotation xmls: " + ", ".join(_.name for _ in candidates) + ". Please provide an annotationsxmlregex to pick the one you want.")
        else:
          raise IOError(f"Found multiple annotation xmls matching {self.__annotationsxmlregex.pattern}: " + ", ".join(_.name for _ in candidates) + ".")
      else:
        if self.__annotationsxmlregex is None:
          return default
        else:
          raise FileNotFoundError(f"Couldn't find any annotation xmls matching {self.__annotationsxmlregex.pattern}")
    if candidate != default:
      self.logger.warning(f"Using {candidate.name} for annotations")
    return candidate

  @property
  def annotationinfofile(self):
    return self.annotationspolygonsxmlfile.with_suffix(".annotationinfo.csv")

class XMLPolygonAnnotationReaderSample(SampleWithAnnotationInfos, XMLPolygonAnnotationReader):
  """
  Base class for any sample that reads the annotations from the XML metadata.
  """
  @property
  def annotationinfofile(self): return self.csv("annotationinfo")

class XMLPolygonAnnotationReaderSampleWithOutline(XMLPolygonAnnotationReaderSample, XMLPolygonAnnotationReaderWithOutline):
  pass

class ReadRectanglesFromXML(ReadRectanglesBase, XMLLayoutReader):
  """
  Base class for any sample that reads rectangles from the XML metadata.
  """
  def readallrectangles(self, **extrakwargs):
    rectangles = self.getrectanglelayout()
    rectangleextrakwargs = self.rectangleextrakwargs
    del rectangleextrakwargs["pscale"]
    return [self.rectangletype(rectangle=r, readingfromfile=False, **rectangleextrakwargs, **extrakwargs) for r in rectangles]

class ReadRectanglesIm3FromXML(ReadRectanglesIm3Base, ReadRectanglesFromXML):
  """
  Base class for any sample that reads rectangles from the XML metadata
  and loads the rectangle images from im3 files.
  """

class ImageCorrectionSample(ImageCorrectionArgumentParser) :
  """
  Base class for any sample that will use corrections defined from input files
  """

  def __init__(self,*args,et_offset_file,skip_et_corrections,flatfield_file,warping_file,correction_model_file,**kwargs) :
    super().__init__(*args,**kwargs)
    self.__et_offset_file = et_offset_file
    self.__skip_et_corrections = skip_et_corrections
    #if the exposure time offset file wasn't given but exposure time corrections aren't supposed to be skipped, then
    #try to get the dark current offset values from a relevant .Full.xml file
    if (not self.__skip_et_corrections) and (self.__et_offset_file is None) :
      if not self.fullxmlfile.is_file() :
        errmsg = f'ERROR: full xml file {self.fullxmlfile} does not exist, please provide a LayerOffset file '
        errmsg+= 'for exposure time dark current offsets or rerun with --skip_exposure_time_corrections'
        raise ValueError(errmsg)
      self.__et_offset_file = self.fullxmlfile
    #set the flatfield/warping files from the arguments
    self.__flatfield_file = flatfield_file
    self.__warping_file = warping_file
    if (correction_model_file is not None) and (self.__flatfield_file is not None or self.__warping_file is not None) :
      warnmsg = f'WARNING: correction model file {correction_model_file} will be ignored because a flatfield and/or '
      warnmsg+= 'warping file were given'
      self.logger.warningonenter(warnmsg) 
    #if neither a flatfield nor a warping file were given, but a correction model file was, search for values defined there
    if self.__flatfield_file is None and self.__warping_file is None and correction_model_file is not None :
      if not correction_model_file.is_file() :
        raise ValueError(f'ERROR: correction model file {correction_model_file} does not exist!')
      table_entries = readtable(correction_model_file,CorrectionModelTableEntry)
      this_slide_tes = [te for te in table_entries if ( te.SlideID==self.SlideID and te.Project==self.Project and 
                                                        te.Cohort==self.Cohort and te.BatchID==self.BatchID ) ]
      if len(this_slide_tes)!=1 :
        errmsg = f'ERROR: there are {len(this_slide_tes)} entries for slide {self.SlideID} in the correction model '
        errmsg+= f'table at {correction_model_file} but there should be exactly one'
        raise RuntimeError(errmsg)
      #reset the warping file
      warping_filename = this_slide_tes[0].WarpingFile
      if warping_filename.lower()=='none' :
        warnmsg = f'WARNING: Warping file for {self.SlideID} in {correction_model_file} is {warping_filename}, '
        warnmsg+= 'warping corrections WILL NOT be applied'
        self.logger.warningonenter(warnmsg)
        self.__warping_file = None
      else :
        self.__warping_file = pathlib.Path(warping_filename)
      #reset the flatfield file
      ff_version = this_slide_tes[0].FlatfieldVersion
      if ff_version.lower()=='none' :
        warnmsg = f'WARNING: Flatfield version for {self.SlideID} in {correction_model_file} is {ff_version}, '
        warnmsg+= 'flatfield corrections WILL NOT be applied'
        self.logger.warningonenter(warnmsg)
        self.__flatfield_file = None
      else :
        ff_filename = f'{UNIV_CONST.FLATFIELD_DIRNAME}_{ff_version}.bin'
        self.__flatfield_file = FF_CONST.DEFAULT_FLATFIELD_MODEL_DIR/ff_filename
    # if the flatfield file argument given isn't a file, 
    # search the astropath_processing/flatfield and root/flatfield directory for a file of the same name
    if (self.__flatfield_file is not None) and (not self.__flatfield_file.is_file()) :
      poss_roots = [self.root,UNIV_CONST.ASTROPATH_PROCESSING_DIR]
      for rd in poss_roots :
        other_filepath = rd / UNIV_CONST.FLATFIELD_DIRNAME / self.__flatfield_file.name
        if other_filepath.is_file() :
          self.__flatfield_file = other_filepath
      if not self.__flatfield_file.is_file() :
        raise ValueError(f'ERROR: flatfield file {self.__flatfield_file} does not exist!')
    # if the warping file argument given isn't a file, 
    # search the astropath_processing/warping and root/warping directories for a file of the same name
    if (self.__warping_file is not None) and (not self.__warping_file.is_file()) :
      poss_roots = [self.root,UNIV_CONST.ASTROPATH_PROCESSING_DIR]
      for rd in poss_roots :
        other_filepath = rd / UNIV_CONST.WARPING_DIRNAME / self.__warping_file.name
        if other_filepath.is_file() :
          self.__warping_file = other_filepath
      if not self.__warping_file.is_file() :
        raise ValueError(f'ERROR: warping file {self.__warping_file} does not exist!')

  @property
  def et_offset_file(self) :
    return self.__et_offset_file
  @property
  def skip_et_corrections(self) :
    return self.__skip_et_corrections
  @property
  def flatfield_file(self) :
    return self.__flatfield_file
  @property
  def warping_file(self) :
    return self.__warping_file
  @property
  def applied_corrections_string(self) :
    corrs = []
    if (not self.__skip_et_corrections) and (self.__et_offset_file is not None) :
      corrs.append('exposure time differences')
    if self.__flatfield_file is not None :
      corrs.append('flatfielding')
    if self.__warping_file is not None :
      corrs.append('warping')
    if len(corrs)==0 :
      return 'not corrected'
    if len(corrs)==1 :
      return f'corrected for {corrs[0]}'
    elif len(corrs)==2 :
      return f'corrected for {corrs[0]} and {corrs[1]}'
    elif len(corrs)==3 :
      return f'corrected for {corrs[0]}, {corrs[1]}, and {corrs[2]}'

class ReadCorrectedRectanglesIm3SingleLayerFromXML(ImageCorrectionSample, ReadRectanglesIm3FromXML) :
  """
  Base class for any sample that reads single layers of rectangles from the XML metadata,
  loads the rectangle images from im3 files, and corrects the rectangle images for differences in exposure time,
  flatfielding effects, and/or warping effects
  """

  multilayer = True #The original files are multilayer, we're just going to be working with one of them
  rectangletype = RectangleCorrectedIm3SingleLayer

  def __init__(self,*args,layer=1,**kwargs) :
    self.__layer = layer
    kwargs['layers']=[self.__layer]
    super().__init__(*args,**kwargs)
    self.__med_et = None

  def initrectangles(self) :
    """
    Init Rectangles with additional transformations for corrections that will be applied
    """
    super().initrectangles()
    #find the median exposure time
    slide_exp_times = np.zeros(shape=(len(self.rectangles)))
    for ir,r in enumerate(self.rectangles) :
        slide_exp_times[ir] = r.allexposuretimes[self.__layer-1]
    self.__med_et = np.median(slide_exp_times)
    if (not self.skip_et_corrections) and (self.et_offset_file is not None) :
      #read the exposure time offsets
      offset = self.__get_exposure_time_offset()
      #add the exposure time correction to every rectangle's transformations
      for r in self.rectangles :
        r.add_exposure_time_correction_transformation(self.__med_et,offset)
    if self.flatfield_file is not None :
      #read the flatfield correction factors from the file
      flatfield = get_raw_as_hwl(self.flatfield_file,
                                 self.rectangles[0].imageshapeinoutput[0],self.rectangles[0].imageshapeinoutput[1],self.nlayers,
                                 np.float64)
      self.logger.infoonenter(f'Flatfield corrections will be applied from {self.flatfield_file}')
      for r in self.rectangles :
        r.add_flatfield_correction_transformation(flatfield[:,:,self.__layer-1])
    if self.warping_file is not None :
      warp = self.__get_warping_object()
      for r in self.rectangles :
        r.add_warping_correction_transformation(warp)

  def __get_exposure_time_offset(self) :
    self.logger.infoonenter(f'Copying exposure time offset for {self.SlideID} layer {self.__layer} from file {self.et_offset_file}')
    #read the offset from the Full.xml file
    if self.et_offset_file==self.fullxmlfile :
      tree = ET.parse(self.et_offset_file)
      root = tree.getroot()
      for child in root :
        if child.tag=='G' :
          for child2 in child.iter() :
            if 'name' in child2.attrib and child2.attrib['name']=='DarkCurrentSettings' :
              for child3 in child2.iter() :
                if 'name' in child3.attrib and child3.attrib['name']=='Mean' :
                  return float(child3.text)
    #read the offset from the LayerOffset file
    else :
      layer_offsets_from_file = readtable(self.et_offset_file,LayerOffset)
      offsets_to_return = [lo.offset for lo in layer_offsets_from_file if lo.layer_n==self.__layer]
      if len(offsets_to_return)!=1 :
        raise ValueError(f'ERROR: found {len(offsets_to_return)} entries for layer {self.__layer} in file {self.et_offset_file}')
      return offsets_to_return[0]

  def __get_warping_object(self) :
    """
    Read a WarpingSummary .csv file and return the CameraWarp object to use for correcting image layers
    """
    warpsummaries = readtable(self.warping_file,WarpingSummary)
    relevant_warps = [ws for ws in warpsummaries if self.__layer in range(ws.first_layer_n,ws.last_layer_n+1)]
    if len(relevant_warps)!=1 :
      raise ValueError(f'ERROR: found {len(relevant_warps)} warps for layer {self.__layer} in {self.warping_file}')
    ws = relevant_warps[0]
    warp = CameraWarp(ws.n,ws.m,ws.cx,ws.cy,ws.fx,ws.fy,ws.k1,ws.k2,ws.k3,ws.p1,ws.p2)
    self.logger.infoonenter(f'Warping corrections will be applied from {self.__warping_file}')
    return warp

  @classmethod
  def makeargumentparser(cls):
    p = super().makeargumentparser()
    p.add_argument('--layer', type=int, default=1,
                   help='The layer number (starting from one) of the images that should be used (default=1)')
    return p
  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
        **super().initkwargsfromargumentparser(parsed_args_dict),
        'layer': parsed_args_dict.pop('layer'),
    }

class ReadCorrectedRectanglesIm3MultiLayerFromXML(ImageCorrectionSample, ReadRectanglesIm3FromXML) :
  """
  Base class for any sample that reads multilayer rectangles from the XML metadata, 
  loads the rectangle images from im3 files, and corrects the rectangle images for differences in exposure time, flatfielding effects, and/or warping
  """

  multilayer = True
  rectangletype = RectangleCorrectedIm3MultiLayer

  def __init__(self,*args,**kwargs) :
    super().__init__(*args,**kwargs)
    self.__med_ets = None

  def initrectangles(self) :
    """
    Init Rectangles with additional transformations for corrections that will be applied
    """
    super().initrectangles()
    #find the median exposure times
    slide_exp_times = np.zeros(shape=(len(self.rectangles),self.nlayers)) 
    for ir,r in enumerate(self.rectangles) :
        slide_exp_times[ir,:] = r.allexposuretimes
    self.__med_ets = np.median(slide_exp_times,axis=0)
    if (not self.skip_et_corrections) and (self.et_offset_file is not None) :
      #read the exposure time offsets
      offsets = self.__read_exposure_time_offsets()
      #add the exposure time correction to every rectangle's transformations
      for r in self.rectangles :
        r.add_exposure_time_correction_transformation(self.__med_ets,offsets)
    if self.flatfield_file is not None :
      #read the flatfield correction factors from the file
      flatfield = get_raw_as_hwl(self.flatfield_file,*(self.rectangles[0].imageshapeinoutput),np.float64)
      self.logger.infoonenter(f'Flatfield corrections will be applied from {self.flatfield_file}')
      for r in self.rectangles :
        r.add_flatfield_correction_transformation(flatfield)
    if self.warping_file is not None :
      warps_by_layer = self.__get_warping_objects_by_layer()
      for r in self.rectangles :
        r.add_warping_correction_transformation(warps_by_layer)

  def __read_exposure_time_offsets(self) :
    """
    Read in the offset factors for exposure time corrections from the file defined by command line args
    """
    self.logger.infoonenter(f'Copying exposure time offsets for {self.SlideID} from file {self.et_offset_file}')
    #read the offset from the Full.xml file
    if self.et_offset_file==self.fullxmlfile :
      tree = ET.parse(self.et_offset_file)
      root = tree.getroot()
      for child in root :
        if child.tag=='G' :
          for child2 in child.iter() :
            if 'name' in child2.attrib and child2.attrib['name']=='DarkCurrentSettings' :
              for child3 in child2.iter() :
                if 'name' in child3.attrib and child3.attrib['name']=='Mean' :
                  to_return = []
                  for li in range(self.nlayers) :
                    to_return.append(float(child3.text))            
                  return to_return
    #read the offsets from the given LayerOffset file
    else :
      layer_offsets_from_file = readtable(self.et_offset_file,LayerOffset)
      offsets_to_return = []
      for ln in range(1,self.nlayers+1) :
          this_layer_offset = [lo.offset for lo in layer_offsets_from_file if lo.layer_n==ln]
          if len(this_layer_offset)==1 :
              offsets_to_return.append(this_layer_offset[0])
          elif len(this_layer_offset)==0 :
              warnmsg = f'WARNING: LayerOffset file {self.et_offset_file} does not have an entry for layer {ln}'
              warnmsg+=  ', so that offset will be set to zero!'
              self.logger.warningonenter(warnmsg)
              offsets_to_return.append(0)
          else :
              raise ValueError(f'ERROR: more than one entry found in LayerOffset file {self.et_offset_file} for layer {ln}!')
      return offsets_to_return

  def __get_warping_objects_by_layer(self) :
    """
    Read a WarpingSummary .csv file and return a list of CameraWarp objects to use for correcting images, one per layer
    """
    warpsummaries = readtable(self.warping_file,WarpingSummary)
    warps_by_layer = []
    for li in range(self.nlayers) :
      warps_by_layer.append(None)
    for ws in warpsummaries :
      if ws.n!=self.rectangles[0].imageshapeinoutput[1] or ws.m!=self.rectangles[0].imageshapeinoutput[0] :
        errmsg = f'ERROR: a warp with dimensions ({ws.m},{ws.n}) cannot be applied to images with '
        errmsg+= f'dimensions ({",".join(self.rectangles[0].imageshapeinoutput[:2])})!'
        raise ValueError(errmsg)
      thiswarp = CameraWarp(ws.n,ws.m,ws.cx,ws.cy,ws.fx,ws.fy,ws.k1,ws.k2,ws.k3,ws.p1,ws.p2)
      for ln in range(ws.first_layer_n,ws.last_layer_n+1) :
        if warps_by_layer[ln-1] is not None :
          raise ValueError(f'ERROR: warping summary {self.warping_file} has conflicting entries for image layer {ln}!')
        warps_by_layer[ln-1] = thiswarp
    self.logger.infoonenter(f'Warping corrections will be applied from {self.warping_file}')
    for li in range(self.nlayers) :
      if warps_by_layer[li] is None :
        warnmsg = f'WARNING: warping summary file {self.warping_file} does not contain any definitions for image layer '
        warnmsg+= f'{li+1} and so warping corrections for this image layer WILL BE SKIPPED!'
        self.logger.warningonenter(warnmsg)
    return warps_by_layer

  @property
  def med_ets(self) :
    if self.et_offset_file is not None and self.__med_ets is None :
      self.initrectangles()
    return self.__med_ets

class ReadRectanglesComponentTiffFromXML(ReadRectanglesComponentTiffBase, ReadRectanglesFromXML):
  """
  Base class for any sample that reads rectangles from the XML metadata
  and loads the rectangle images from component tiff files.
  """

class ReadRectanglesOverlapsFromXML(ReadRectanglesOverlapsBase, ReadRectanglesFromXML):
  """
  Base class for any sample that reads rectangles and overlaps from the XML metadata.
  """
  def readalloverlaps(self, **kwargs):
    return self.getoverlaps(overlaptype=self.overlaptype, **kwargs)

class ReadRectanglesOverlapsIm3FromXML(ReadRectanglesOverlapsIm3Base, ReadRectanglesOverlapsFromXML, ReadRectanglesIm3FromXML):
  """
  Base class for any sample that reads rectangles and overlaps from the XML metadata
  and loads the rectangle images from im3 files.
  """

class ReadCorrectedRectanglesOverlapsIm3SingleLayerFromXML(ReadRectanglesOverlapsIm3Base, ReadRectanglesOverlapsFromXML, ReadCorrectedRectanglesIm3SingleLayerFromXML) :
  """
  Base class for any sample that reads a single layer of corrected multilayer rectangles and also reads overlaps from XML metadata 
  """

class ReadCorrectedRectanglesOverlapsIm3MultiLayerFromXML(ReadRectanglesOverlapsIm3Base, ReadRectanglesOverlapsFromXML, ReadCorrectedRectanglesIm3MultiLayerFromXML) :
  """
  Base class for any sample that reads corrected multilayer rectangles and also reads overlaps from XML metadata 
  """

class ReadRectanglesOverlapsComponentTiffFromXML(ReadRectanglesOverlapsComponentTiffBase, ReadRectanglesOverlapsFromXML, ReadRectanglesComponentTiffFromXML):
  """
  Base class for any sample that reads rectangles and overlaps from the XML metadata
  and loads the rectangle images from component tiff files.
  """

class TempDirSample(SampleBase, TempDirArgumentParser):
  """
  Base class for any sample that wants to use a temp folder
  temproot: main folder to make the tmpdir in (default is whatever the system uses)
  """
  def __init__(self, *args, temproot=None, **kwargs):
    if temproot is not None: temproot = pathlib.Path(temproot)
    self.__temproot = temproot
    super().__init__(*args, **kwargs)

  @methodtools.lru_cache()
  @property
  def tempfolder(self):
    return self.enter_context(tempfile.TemporaryDirectory(dir=self.__temproot))

  def tempfile(self, *args, **kwargs):
    return self.enter_context(tempfile.NamedTemporaryFile(*args, dir=self.tempfolder, **kwargs))

class ParallelSample(SampleBase, ParallelArgumentParser):
  """
  Base class for any sample that runs jobs in parallel
  njobs: maximum number of jobs to use (default is no maximum)
  """
  def __init__(self, *args, njobs=None, **kwargs):
    self.__njobs = njobs
    super().__init__(*args, **kwargs)

  @property
  def njobs(self):
    return self.__njobs
  def pool(self):
    nworkers = mp.cpu_count()
    if self.njobs is not None: nworkers = min(nworkers, self.njobs)
    return mp.get_context().Pool(nworkers)

class SampleWithSegmentations(SampleBase):
  @classmethod
  @abc.abstractmethod
  def segmentationalgorithm(cls): pass

class InformSegmentationSample(SampleWithSegmentations):
  @classmethod
  def segmentationalgorithm(cls):
    return "inform"

  @methodtools.lru_cache()
  @property
  def segmentationids(self):
    dct = {}
    for layer in self.mergeconfig:
      segstatus = layer.SegmentationStatus
      if segstatus != 0:
        segid = layer.ImageQA
        if segid == "NA":
          segid = segstatus
        if segstatus not in dct:
          dct[segstatus] = segid
        elif segid != segstatus:
          if segid != dct[segstatus] != segstatus:
            raise ValueError(f"Multiple different non-NA ImageQAs for SegmentationStatus {segstatus} ({self.mergeconfigcsv})")
          else:
            dct[segstatus] = segid
    if sorted(dct.keys()) != list(range(1, len(dct)+1)):
      raise ValueError(f"Non-sequential SegmentationStatuses {sorted(dct.keys())} ({self.mergeconfigcsv})")
    return tuple(dct[k] for k in range(1, len(dct)+1))

  @property
  def nsegmentations(self):
    return len(self.segmentationids)

class ReadRectanglesSegmentedComponentTiffBase(ReadRectanglesComponentTiffBase, InformSegmentationSample):
  @property
  def with_seg(self): return True

  @property
  def masklayer(self):
    return self.nlayersunmixed + 1
  def segmentationnucleuslayer(self, segid):
    return self.nlayersunmixed + 2 + self.segmentationids.index(segid)
  def segmentationmembranelayer(self, segid):
    return self.nlayersunmixed + 2 + self.nsegmentations + self.segmentationids.index(segid)

  def isnucleuslayer(self, layer):
    return self.nlayersunmixed + 1 < layer <= self.nlayersunmixed + 1 + self.nsegmentations
  def ismembranelayer(self, layer):
    return self.nlayersunmixed + 1 + self.nsegmentations < layer <= self.nlayersunmixed + 1 + 2*self.nsegmentations
  def segmentationidfromlayer(self, layer):
    if layer <= self.masklayer: raise ValueError(f"{layer} is not a segmentation layer")
    idx = layer - self.masklayer
    if self.ismembranelayer(layer):
      idx -= self.nsegmentations
    return self.segmentationids[idx-1]

  @property
  def rectangleextrakwargs(self):
    kwargs = {
      **super().rectangleextrakwargs,
      "nsegmentations": self.nsegmentations
    }
    return kwargs

class ReadRectanglesDbloadSegmentedComponentTiff(ReadRectanglesDbloadComponentTiff, ReadRectanglesSegmentedComponentTiffBase):
  pass
