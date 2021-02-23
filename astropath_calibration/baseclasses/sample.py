import abc, contextlib, cv2, dataclassy, datetime, fractions, functools, itertools, jxmlease, logging, methodtools, numpy as np, os, pathlib, re, tempfile, tifffile

from ..utilities import units
from ..utilities.dataclasses import MyDataClass
from ..utilities.misc import floattoint
from ..utilities.tableio import readtable, writetable
from .annotationxmlreader import AnnotationXMLReader
from .csvclasses import constantsdict, RectangleFile
from .logging import getlogger
from .rectangle import Rectangle, RectangleCollection, rectangleoroverlapfilter, RectangleReadComponentTiff, RectangleReadComponentTiffMultiLayer, RectangleReadIm3, RectangleReadIm3MultiLayer
from .overlap import Overlap, OverlapCollection, RectangleOverlapCollection

class SampleDef(MyDataClass):
  """
  The sample definition from sampledef.csv in the cohort folder.
  To construct it, you can give all the arguments, or you can give
  SlideID and leave out some of the others.  If you give a root,
  it will try to figure out the other arguments from there.
  """
  SampleID: int
  SlideID: str
  Project: int = None
  Cohort: int = None
  Scan: int = None
  BatchID: int = None
  isGood: int = True

  @classmethod
  def transforminitargs(cls, *args, root=None, samp=None, **kwargs):
    if samp is not None:
      if isinstance(samp, str):
        if "SlideID" in kwargs:
          raise TypeError("Provided both samp and SlideID")
        else:
          kwargs["SlideID"] = samp
      else:
        if args or kwargs:
          raise TypeError("Have to give either a sample or other arguments, not both.")
        return super().transforminitargs(*args, **kwargs, **{field: getattr(samp, field) for field in dataclassy.fields(SampleDef)})

    if "SlideID" in kwargs and root is not None:
      root = pathlib.Path(root)
      try:
        cohorttable = readtable(root/"sampledef.csv", SampleDef)
      except IOError:
        pass
      else:
        for row in cohorttable:
          if row.SlideID == kwargs["SlideID"]:
            return cls.transforminitargs(root=root, samp=row)

      if "Scan" not in kwargs:
        try:
          kwargs["Scan"] = max(int(folder.name.replace("Scan", "")) for folder in (root/kwargs["SlideID"]/"im3").glob("Scan*/"))
        except ValueError:
          pass
      if "BatchID" not in kwargs and kwargs.get("Scan", None) is not None:
        try:
          with open(root/kwargs["SlideID"]/"im3"/f"Scan{kwargs['Scan']}"/"BatchID.txt") as f:
            kwargs["BatchID"] = int(f.read())
        except FileNotFoundError:
          pass

    if "SampleID" not in kwargs: kwargs["SampleID"] = 0

    return super().transforminitargs(*args, **kwargs)

  def __bool__(self):
    return bool(self.isGood)

class SampleBase(contextlib.ExitStack, units.ThingWithPscale):
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
  def __init__(self, root, samp, *, xmlfolders=None, uselogfiles=False, logthreshold=logging.DEBUG, reraiseexceptions=True, logroot=None, mainlog=None, samplelog=None):
    self.root = pathlib.Path(root)
    self.samp = SampleDef(root=root, samp=samp)
    if not (self.root/self.SlideID).exists():
      raise IOError(f"{self.root/self.SlideID} does not exist")
    if logroot is None: logroot = root
    self.logroot = pathlib.Path(logroot)
    self.logger = getlogger(module=self.logmodule, root=self.logroot, samp=self.samp, uselogfiles=uselogfiles, threshold=logthreshold, reraiseexceptions=reraiseexceptions, mainlog=mainlog, samplelog=samplelog)
    if xmlfolders is None: xmlfolders = []
    self.__xmlfolders = xmlfolders
    self.__logonenter = []
    self.__entered = False
    super().__init__()

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

  def __str__(self):
    return str(self.mainfolder)

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
    return self.mainfolder/"im3"

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
    return self.mainfolder/"inform_data"/"Component_Tiffs"

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
    return self.__xmlfolders + [self.im3folder/"xml"]
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

  def __getimageinfofromXMLfiles(self):
    """
    Find the pscale, image dimensions, and number of layers from the XML metadata.
    """
    with open(self.xmlfolder/(self.SlideID+".Parameters.xml"), "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="/IM3Fragment/D"):
        if node.xml_attrs["name"] == "Shape":
          width, height, nlayers = (int(_) for _ in str(node).split())
        if node.xml_attrs["name"] == "MillimetersPerPixel":
          pscale = 1e-3/float(node)

    try:
      width *= units.onepixel(pscale=pscale)
      height *= units.onepixel(pscale=pscale)
    except NameError:
      raise IOError(f'Couldn\'t find Shape and/or MillimetersPerPixel in {self.xmlfolder/(self.SlideID+".Parameters.xml")}')

    return pscale, width, height, nlayers

  @methodtools.lru_cache()
  @property
  def samplelocation(self):
    """
    Find the location of the image on the slide from the xml metadata
    """
    with open(self.xmlfolder/(self.SlideID+".Parameters.xml"), "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="/IM3Fragment/D"):
        if node.xml_attrs["name"] == "SampleLocation":
          return np.array([float(_) for _ in str(node).split()]) * units.onemicron(pscale=self.apscale)

  @methodtools.lru_cache()
  def __getcamerastateparameter(self, parametername):
    """
    Find info from the CameraState Parameters block in the xml metadata
    """
    with open(self.xmlfolder/(self.SlideID+".Full.xml"), "rb") as f:
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
    with open(self.xmlfolder/(self.SlideID+".Full.xml"), "rb") as f:
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
    with open(self.xmlfolder/(self.SlideID+".Parameters.xml"), "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="/IM3Fragment/D"):
        if node.xml_attrs["name"] == "Shape":
          width, height, nlayers = (int(_) for _ in str(node).split())
          return nlayers
      else:
        raise IOError(f'Couldn\'t find Shape in {self.xmlfolder/(self.SlideID+".Parameters.xml")}')

  @methodtools.lru_cache()
  @property
  def nlayersunmixed(self):
    """
    Find the number of component tiff layers from the xml metadata
    """
    with open(self.componenttiffsfolder/"batch_procedure.ifp", "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="AllComponents"):
        return int(node.xml_attrs["dim"])

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
        if warnfunction != self.logger.warningglobal: warnfunction = self.logger.warning
      else:
        warnfunction = self.logger.warningglobal

    if warnfunction is not None:
      if not self.logger.nentered:
        warnfunction = functools.partial(self._logonenter, warnfunction=warnfunction)
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

  def _logonenter(self, warning, warnfunction):
    """
    Puts the function and warning into a queue to be logged
    when we enter the with statement.
    """
    self.__logonenter.append((warnfunction, warning))

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
    xmlfile = self.scanfolder/(self.SlideID+"_"+self.scanfolder.name+"_annotations.xml")
    reader = AnnotationXMLReader(xmlfile, pscale=self.pscale)
    return reader.rectangles, reader.globals, reader.perimeters, reader.microscopename

  @property
  def microscopename(self):
    """
    Name of the computer that processed the image.
    """
    return self.getXMLplan()[3]

  def __enter__(self):
    self.__entered = True
    self.enter_context(self.logger)
    for warnfunction, warning in self.__logonenter:
      warnfunction(warning)
    return super().__enter__()

  #def enter_context(self, *args, **kwargs):
  #  if not self.__entered:
  #    raise ValueError(f"Have to use {self} in a with statement if you want to enter_context")
  #  return super().enter_context(*args, **kwargs)

  @abc.abstractproperty
  def logmodule(self):
    "name of the log files for this class (e.g. align)"

class DbloadSampleBase(SampleBase):
  """
  Base class for any sample that uses the csvs in the dbload folder.
  If the dbload folder is initialized already, you should instead inherit
  from DbloadSample to get more functionality.

  dbloadfolder: Folder to look for the csv files in (default: dbloadroot/SlideID/dbload)
  dbloadroot: A different root to use to find the dbload (default: same as root)
  """
  def __init__(self, *args, dbloadroot=None, dbloadfolder=None, **kwargs):
    super().__init__(*args, **kwargs)
    if dbloadroot is not None and dbloadfolder is not None:
      raise TypeError("Can't provide both dbloadroot and dbloadfolder")
    if dbloadroot is None:
      dbloadroot = self.mainfolder.parent
    else:
      dbloadroot = pathlib.Path(dbloadroot)
    if dbloadfolder is None:
      dbloadfolder = dbloadroot/self.SlideID/"dbload"
    else:
      dbloadfolder = pathlib.Path(dbloadfolder)
    self.__dbloadfolder = dbloadfolder
  @property
  def dbload(self):
    """
    The folder where the csv files live.
    """
    return self.__dbloadfolder

  def csv(self, csv):
    """
    Full path to the csv file identified by csv.
    """
    return self.dbload/f"{self.SlideID}_{csv}.csv"
  def readcsv(self, csv, *args, **kwargs):
    """
    Read the csv file using readtable.
    """
    return readtable(self.csv(csv), *args, **kwargs)
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
    return cv2.imread(os.fspath(self.dbload/(self.SlideID+"_qptiff.jpg")))

class DbloadSample(DbloadSampleBase, units.ThingWithQpscale, units.ThingWithApscale):
  """
  Base class for any sample that uses the csvs in the dbload folder
  after the folder has been set up.

  dbloadfolder: Folder where the csv files live (default: dbloadroot/SlideID/dbload)
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

class Im3SampleBase(SampleBase):
  """
  Base class for any sample that uses sharded im3 images.
  root2: Root location of the im3 images.
         (The images are in root2/SlideID)
  """
  def __init__(self, root, root2, samp, *args, **kwargs):
    super().__init__(root=root, samp=samp, *args, **kwargs)
    self.root2 = pathlib.Path(root2)

  @property
  def root1(self): return self.root

  @property
  def possiblexmlfolders(self):
    return super().possiblexmlfolders + [self.root2/self.SlideID]

class ZoomSampleBase(SampleBase):
  """
  Base class for any sample that uses the zoomed "big" or "wsi" images.
  zoomroot: Root location of the zoomed images.
            (The images are in zoomroot/SlideID/big and zoomroot/SlideID/wsi)
  """
  def __init__(self, *args, zoomroot, **kwargs):
    super().__init__(*args, **kwargs)
    self.__zoomroot = pathlib.Path(zoomroot)
  @property
  def zoomroot(self): return self.__zoomroot
  @property
  def zoomfolder(self): return self.zoomroot/self.SlideID/"big"
  @property
  def wsifolder(self): return self.zoomroot/self.SlideID/"wsi"

  @property
  def zmax(self): return 9
  def zoomfilename(self, layer, tilex, tiley):
    """
    Zoom filename for a given layer and tile.
    """
    return self.zoomfolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-X{tilex}-Y{tiley}-big.png"
  def wsifilename(self, layer):
    """
    Wsi filename for a given layer.
    """
    return self.wsifolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-wsi.png"

class DeepZoomSampleBase(SampleBase):
  """
  Base class for any sample that uses the deepzoomed images.
  deepzoomroot: Root location of the deepzoomed images.
                (The images are in deepzoomroot/SlideID)
  """
  def __init__(self, *args, deepzoomroot, **kwargs):
    super().__init__(*args, **kwargs)
    self.__deepzoomroot = pathlib.Path(deepzoomroot)
  @property
  def deepzoomroot(self): return self.__deepzoomroot
  @property
  def deepzoomfolder(self): return self.deepzoomroot/self.SlideID

class GeomSampleBase(SampleBase):
  """
  Base class for any sample that uses the _cellgeomload.csv files

  geomfolder: Folder where the _cellgeomload.csv files live (default: geomroot/SlideID/geom)
  geomroot: A different root to use to find the cellgeomload (default: same as root)
  """
  def __init__(self, *args, geomroot=None, geomfolder=None, **kwargs):
    if geomroot is not None and geomfolder is not None:
      raise TypeError("Can't provide both geomroot and geomfolder")
    self.__geomroot = geomroot
    self.__geomfolder = geomfolder
    super().__init__(*args, **kwargs)

  @methodtools.lru_cache()
  @property
  def geomfolder(self):
    geomroot = self.__geomroot
    geomfolder = self.__geomfolder
    if geomfolder is None:
      if geomroot is None:
        geomroot = self.mainfolder.parent
      else:
        geomroot = pathlib.Path(geomroot)
      return geomroot/self.SlideID/"geom"
    else:
      return pathlib.Path(geomfolder)

class ReadRectanglesBase(RectangleCollection, SampleBase):
  """
  Base class for any sample that reads HPF info from any source.
  selectrectangles: filter for selecting rectangles (a list of ids or a function)
  """
  def __init__(self, *args, selectrectangles=None, **kwargs):
    super().__init__(*args, **kwargs)
    rectanglefilter = rectangleoroverlapfilter(selectrectangles)
    self.__rectangles  = self.readallrectangles()
    self.__rectangles = [r for r in self.rectangles if rectanglefilter(r)]

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
    return kwargs

  @property
  def _rectangles(self):
    assert False
  @_rectangles.setter
  def _rectangles(self, value):
    self.__rectangles = value

  @property
  def rectangles(self): return self.__rectangles

class ReadRectanglesWithLayers(ReadRectanglesBase):
  """
  Base class for any sample that reads rectangles
  and needs a layer selection.
  """
  def __init__(self, *args, layer=None, layers=None, **kwargs):
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

    super().__init__(*args, **kwargs)

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

  multilayer = False #can override in subclasses

class ReadRectanglesIm3Base(ReadRectanglesWithLayers, Im3SampleBase):
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

    if self.multilayer:
      if readlayerfile:
        raise ValueError(f"Can't read a layer file for a multilayer sample {type(self).__name__}")
    else:
      self.__readlayerfile = readlayerfile

    super().__init__(*args, **kwargs)

  @property
  def nlayers(self):
    return self.nlayersim3
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
      "imagefolder": self.root2/self.SlideID,
      "filetype": self.filetype,
      "width": self.fwidth,
      "height": self.fheight,
    }
    try:
      kwargs.update({
        "xmlfolder": self.xmlfolder,
      })
    except FileNotFoundError:
      pass
    if self.multilayer:
      pass
    else:
      kwargs.update({
        "readlayerfile": self.__readlayerfile,
      })
    return kwargs

  @property
  def filetype(self): return self.__filetype

class ReadRectanglesComponentTiffBase(ReadRectanglesWithLayers):
  """
  Base class for any sample that loads images from a component tiff file.
  layer or layers: the layer or layers to read, depending on whether
                   the class uses multilayer images or not
  with_seg: whether or not to read from the _w_seg component tiff file
  """

  def __init__(self, *args, with_seg=False, **kwargs):
    self.__with_seg = with_seg
    super().__init__(*args, **kwargs)

  @property
  def nlayers(self):
    return self.nlayersunmixed
  @property
  def rectangletype(self):
    if self.multilayer:
      return RectangleReadComponentTiffMultiLayer
    else:
      return RectangleReadComponentTiff
  @property
  def rectangleextrakwargs(self):
    kwargs = {
      **super().rectangleextrakwargs,
      "imagefolder": self.componenttiffsfolder,
      "with_seg": self.__with_seg,
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
    overlapfilter = lambda o: _overlapfilter(o) and o.p1 in self.rectangleindices and o.p2 in self.rectangleindices

    self.__overlaps  = self.readalloverlaps()
    self.__overlaps = [o for o in self.overlaps if overlapfilter(o)]
    if onlyrectanglesinoverlaps:
      self._rectangles = [r for r in self.rectangles if self.selectoverlaprectangles(r)]

  @abc.abstractmethod
  def readalloverlaps(self): pass
  @property
  def overlapextrakwargs(self):
    return {"pscale": self.pscale, "rectangles": self.rectangles, "nclip": self.nclip}

  @property
  def overlaps(self): return self.__overlaps

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

class ReadRectanglesDbload(ReadRectanglesBase, DbloadSample):
  """
  Base class for any sample that reads rectangles from the dbload folder.
  """
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
    rectanglefiles = self.getdir()
    maxtimediff = datetime.timedelta(0)
    for r in rectangles:
      rfs = {rf for rf in rectanglefiles if np.all(rf.cxvec == r.cxvec)}
      assert len(rfs) <= 1
      if not rfs:
        cx, cy = floattoint(r.cxvec / self.onemicron, atol=1e-10)
        errormessage = f"File {self.SlideID}_[{cx},{cy}].im3 (expected from annotations) does not exist"
        if self.__checkim3s:
          raise FileNotFoundError(errormessage)
        else:
          self.logger.warning(errormessage)
      else:
        rf = rfs.pop()
        maxtimediff = max(maxtimediff, abs(rf.t-r.t))
    if maxtimediff >= datetime.timedelta(seconds=5):
      self.logger.warning(f"Biggest time difference between annotation and file mtime is {maxtimediff}")
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
        self.logger.warningglobal(f"{rectangle.file} has _M2 in the name.  {len(duplicates)} other duplicate rectangles.")
    for i, rectangle in enumerate(rectangles, start=1):
      rectangle.n = i

  def fixrectanglefilenames(self, rectangles):
    """
    Fix rectangle filenames if the coordinates are messed up
    """
    for r in rectangles:
      expected = self.SlideID+f"_[{floattoint(r.cx/r.onemicron, atol=1e-10):d},{floattoint(r.cy/r.onemicron, atol=1e-10):d}].im3"
      actual = r.file
      if expected != actual:
        self.logger.warningglobal(f"rectangle at ({r.cx}, {r.cy}) has the wrong filename {actual}.  Changing it to {expected}.")
      r.file = expected

  @methodtools.lru_cache()
  def getdir(self):
    """
    List all rectangles that have im3 files.
    """
    folder = self.scanfolder/"MSI"
    im3s = folder.glob("*.im3")
    result = []
    for im3 in im3s:
      regex = self.SlideID+r"_\[([0-9]+),([0-9]+)\].im3"
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

class ReadRectanglesOverlapsComponentTiffFromXML(ReadRectanglesOverlapsComponentTiffBase, ReadRectanglesOverlapsFromXML, ReadRectanglesComponentTiffFromXML):
  """
  Base class for any sample that reads rectangles and overlaps from the XML metadata
  and loads the rectangle images from component tiff files.
  """

class TempDirSample(SampleBase):
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
