import abc, contextlib, dataclassy, datetime, fractions, functools, itertools, jxmlease, logging, methodtools, numpy as np, os, pathlib, re, tempfile, tifffile

from ..utilities import units
from ..utilities.dataclasses import MyDataClass
from ..utilities.misc import floattoint
from ..utilities.tableio import readtable, writetable
from .annotationxmlreader import AnnotationXMLReader
from .csvclasses import constantsdict, RectangleFile
from .logging import getlogger
from .rectangle import Rectangle, RectangleCollection, rectangleoroverlapfilter, RectangleReadComponentTiff, RectangleReadComponentTiffMultiLayer, RectangleWithImage, RectangleWithImageMultiLayer
from .overlap import Overlap, RectangleOverlapCollection

class SampleDef(MyDataClass):
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
  def __init__(self, root, samp, *, uselogfiles=False, logthreshold=logging.DEBUG, xmlfolders=None, reraiseexceptions=True, logroot=None, mainlog=None, samplelog=None):
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
    return self.root/self.SlideID

  @property
  def im3folder(self):
    return self.mainfolder/"im3"

  @property
  def scanfolder(self):
    return self.im3folder/f"Scan{self.Scan}"

  @property
  def qptifffilename(self):
    return self.scanfolder/(self.SlideID+"_"+self.scanfolder.name+".qptiff")

  @property
  def componenttiffsfolder(self):
    return self.mainfolder/"inform_data"/"Component_Tiffs"

  def getimageinfofromcomponenttiff(self):
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
    return self.__xmlfolders + [self.im3folder/"xml"]
  @property
  def xmlfolder(self):
    possibilities = self.possiblexmlfolders
    for possibility in possibilities:
      if (possibility/(self.SlideID+".Parameters.xml")).exists(): return possibility
    raise FileNotFoundError(f"Didn't find {self.SlideID}.Parameters.xml in any of these folders:\n" + ", ".join(str(_) for _ in possibilities))

  def getimageinfofromXMLfiles(self):
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
    with open(self.xmlfolder/(self.SlideID+".Parameters.xml"), "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="/IM3Fragment/D"):
        if node.xml_attrs["name"] == "SampleLocation":
          return np.array([float(_) for _ in str(node).split()]) * units.onemicron(pscale=self.apscale)

  @methodtools.lru_cache()
  def getcamerastateparameter(self, parametername):
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
    return self.getcamerastateparameter("ResolutionBits")

  @property
  def gainfactor(self):
    return self.getcamerastateparameter("GainFactor")

  @methodtools.lru_cache()
  def getcamerabinning(self, xory):
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
  def camerabinningx(self): return self.getcamerabinning("X")
  @property
  def camerabinningy(self): return self.getcamerabinning("Y")

  @methodtools.lru_cache()
  @property
  def nlayersim3(self):
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
    with open(self.componenttiffsfolder/"batch_procedure.ifp", "rb") as f:
      for path, _, node in jxmlease.parse(f, generator="AllComponents"):
        return int(node.xml_attrs["dim"])

  def getimageinfos(self):
    result = {}
    try:
      result["component tiff"] = self.getimageinfofromcomponenttiff()
    except FileNotFoundError:
      result["component tiff"] = None
    try:
      result["xml files"] = self.getimageinfofromXMLfiles()
    except (FileNotFoundError, IOError):
      result["xml files"] = None
    return result

  @methodtools.lru_cache()
  def getimageinfo(self):
    results = self.getimageinfos()
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
        warnfunction = functools.partial(self.logonenter, warnfunction=warnfunction)
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

  def logonenter(self, warning, warnfunction):
    self.__logonenter.append((warnfunction, warning))

  @property
  def pscale(self): return self.getimageinfo()[0]
  @property
  def fwidth(self): return self.getimageinfo()[1]
  @property
  def fheight(self): return self.getimageinfo()[2]
  @property
  def flayers(self):
    layers = self.getimageinfo()[3]
    if layers is None: raise FileNotFoundError("Couldn't get image info from any source that has flayers")
    return layers

  @methodtools.lru_cache()
  def getXMLplan(self):
    xmlfile = self.scanfolder/(self.SlideID+"_"+self.scanfolder.name+"_annotations.xml")
    reader = AnnotationXMLReader(xmlfile, pscale=self.pscale)
    return reader.rectangles, reader.globals, reader.perimeters, reader.microscopename

  @property
  def microscopename(self):
    return self.getXMLplan()[3]

  def __enter__(self):
    self.__entered = True
    self.enter_context(self.logger)
    for warnfunction, warning in self.__logonenter:
      warnfunction(warning)
    return super().__enter__()

  def enter_context(self, *args, **kwargs):
    if not self.__entered:
      raise ValueError(f"Have to use {self} in a with statement if you want to enter_context")
    super().enter_context(*args, **kwargs)

  @abc.abstractproperty
  def logmodule(self):
    "name of the log files for this class (e.g. align)"

class DbloadSampleBase(SampleBase):
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
    return self.__dbloadfolder

  def csv(self, csv):
    return self.dbload/f"{self.SlideID}_{csv}.csv"
  def readcsv(self, csv, *args, **kwargs):
    return readtable(self.csv(csv), *args, **kwargs)
  def writecsv(self, csv, *args, **kwargs):
    return writetable(self.csv(csv), *args, logger=self.logger, **kwargs)

class DbloadSample(DbloadSampleBase, units.ThingWithQpscale, units.ThingWithApscale):
  def getimageinfofromconstants(self):
    dct = constantsdict(self.csv("constants"))

    fwidth    = dct["fwidth"]
    fheight   = dct["fheight"]
    flayers   = dct["flayers"]
    pscale    = float(dct["pscale"])

    return pscale, fwidth, fheight, flayers

  def getimageinfos(self):
    result = super().getimageinfos()

    try:
      result["constants.csv"] = self.getimageinfofromconstants()
    except (FileNotFoundError, KeyError):
      result["constants.csv"] = None

    return result

  @methodtools.lru_cache()
  @property
  def constantsdict(self):
    return constantsdict(self.csv("constants"), pscale=self.pscale)

  @property
  def position(self):
    return np.array([self.constantsdict["xposition"], self.constantsdict["yposition"]])
  @property
  def nclip(self):
    return self.constantsdict["nclip"]
  @property
  def qpscale(self):
    return self.constantsdict["qpscale"]
  @property
  def apscale(self):
    return self.constantsdict["apscale"]

class FlatwSampleBase(SampleBase):
  def __init__(self, root, root2, samp, *args, root3=None, xmlfolders=None, **kwargs):
    if xmlfolders is None: xmlfolders = []
    super().__init__(root=root, samp=samp, *args, xmlfolders=xmlfolders, **kwargs)
    self.root2 = pathlib.Path(root2)
    self.__root3 = pathlib.Path(root3) if root3 is not None else root3

  @property
  def root1(self): return self.root

  @property
  def possiblexmlfolders(self):
    result = super().possiblexmlfolders + [self.root2/self.SlideID]
    if self.__root3 is not None:
      result.append(self.__root3/self.SlideID)
    return result

class ZoomSampleBase(SampleBase):
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
    return self.zoomfolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-X{tilex}-Y{tiley}-big.png"
  def wsifilename(self, layer):
    return self.wsifolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-wsi.png"

class DeepZoomSampleBase(SampleBase):
  def __init__(self, *args, deepzoomroot, **kwargs):
    super().__init__(*args, **kwargs)
    self.__deepzoomroot = pathlib.Path(deepzoomroot)
  @property
  def deepzoomroot(self): return self.__deepzoomroot
  @property
  def deepzoomfolder(self): return self.deepzoomroot/self.SlideID

class GeomSampleBase(SampleBase):
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

class SampleThatReadsRectangles(SampleBase):
  rectangletype = Rectangle #can be overridden in subclasses

class SampleThatReadsOverlaps(SampleThatReadsRectangles):
  overlaptype = Overlap #can be overridden in subclasses

class ReadRectanglesBase(SampleThatReadsRectangles, RectangleCollection):
  @abc.abstractmethod
  def readallrectangles(self): pass
  @abc.abstractproperty
  def rectangletype(self): pass
  @abc.abstractproperty
  def nlayers(self): pass
  @property
  def rectangleextrakwargs(self):
    kwargs = {
      "pscale": self.pscale,
    }
    return kwargs

  def __init__(self, *args, selectrectangles=None, layers=None, **kwargs):
    super().__init__(*args, **kwargs)
    if layers is None:
      layers = range(1, self.nlayers+1)
    self.__layers = layers
    rectanglefilter = rectangleoroverlapfilter(selectrectangles)
    self.__rectangles  = self.readallrectangles()
    self.__rectangles = [r for r in self.rectangles if rectanglefilter(r)]

  @property
  def _rectangles(self):
    assert False
  @_rectangles.setter
  def _rectangles(self, value):
    self.__rectangles = value

  @property
  def rectangles(self): return self.__rectangles

  @property
  def layers(self): return self.__layers

class ReadRectanglesIm3Base(ReadRectanglesBase, FlatwSampleBase):
  @property
  def nlayers(self):
    return self.nlayersim3
  @property
  def rectangletype(self):
    if self.multilayer:
      return RectangleWithImageMultiLayer
    else:
      return RectangleWithImage
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
      kwargs.update({
        "layers": self.layers,
        "nlayers": self.nlayers,
      })
    else:
      kwargs.update({
        "layer": self.layer,
        "readlayerfile": self.__readlayerfile,
      })
      if not self.__readlayerfile:
        kwargs["nlayers"] = self.nlayers
    return kwargs

  @property
  def filetype(self): return self.__filetype

  multilayer = False #can override in subclasses

  def __init__(self, *args, filetype, layer=None, layers=None, readlayerfile=True, **kwargs):
    self.__filetype = filetype

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
    self.__readlayerfile = readlayerfile

    super().__init__(*args, layers=layers, **kwargs)

  @property
  def layer(self):
    if self.multilayer:
      raise TypeError(f"Can't get layer for a multilayer sample {type(self).__name__}")
    return self.__layer

class ReadRectanglesComponentTiffBase(ReadRectanglesBase):
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

  multilayer = True #can override in subclasses

  def __init__(self, *args, layer=None, layers=None, with_seg=False, **kwargs):
    self.__with_seg = with_seg
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

    super().__init__(*args, layers=layers, **kwargs)

  @property
  def layer(self):
    if self.multilayer:
      raise TypeError(f"Can't get layer for a multilayer sample {type(self).__name__}")
    return self.__layer

class ReadRectanglesOverlapsBase(ReadRectanglesBase, SampleThatReadsOverlaps, RectangleOverlapCollection):
  @abc.abstractmethod
  def readalloverlaps(self): pass
  @property
  def overlapextrakwargs(self):
    return {"pscale": self.pscale, "rectangles": self.rectangles, "nclip": self.nclip}

  multilayer = False #can override in subclasses

  def __init__(self, *args, selectoverlaps=None, onlyrectanglesinoverlaps=False, **kwargs):
    super().__init__(*args, **kwargs)

    _overlapfilter = rectangleoroverlapfilter(selectoverlaps)
    overlapfilter = lambda o: _overlapfilter(o) and o.p1 in self.rectangleindices and o.p2 in self.rectangleindices

    self.__overlaps  = self.readalloverlaps()
    self.__overlaps = [o for o in self.overlaps if overlapfilter(o)]
    if onlyrectanglesinoverlaps:
      self._rectangles = [r for r in self.rectangles if self.selectoverlaprectangles(r)]

  @property
  def overlaps(self): return self.__overlaps

class ReadRectanglesOverlapsIm3Base(ReadRectanglesOverlapsBase, ReadRectanglesIm3Base):
  pass

class ReadRectangles(ReadRectanglesBase, DbloadSample):
  @property
  def rectanglecsv(self): return "rect"
  def readallrectangles(self, **extrakwargs):
    return self.readcsv(self.rectanglecsv, self.rectangletype, extrakwargs={**self.rectangleextrakwargs, **extrakwargs})

class ReadRectanglesIm3(ReadRectangles, ReadRectanglesIm3Base):
  pass

class ReadRectanglesComponentTiff(ReadRectangles, ReadRectanglesComponentTiffBase):
  pass

class ReadRectanglesOverlaps(ReadRectangles, ReadRectanglesOverlapsBase):
  @property
  def overlapcsv(self): return "overlap"
  def readalloverlaps(self, *, overlaptype=None, **extrakwargs):
    if overlaptype is None: overlaptype = self.overlaptype
    return self.readcsv(self.overlapcsv, overlaptype, filter=lambda row: row["p1"] in self.rectangleindices and row["p2"] in self.rectangleindices, extrakwargs={**self.overlapextrakwargs, **extrakwargs})

class ReadRectanglesOverlapsIm3(ReadRectanglesOverlaps, ReadRectanglesOverlapsIm3Base):
  pass

class XMLLayoutReader(SampleThatReadsOverlaps):
  def __init__(self, *args, checkim3s=False, **kwargs):
    self.__checkim3s = checkim3s
    super().__init__(*args, **kwargs)

  @methodtools.lru_cache()
  def getrectanglelayout(self):
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
    for r in rectangles:
      expected = self.SlideID+f"_[{floattoint(r.cx/r.onemicron, atol=1e-10):d},{floattoint(r.cy/r.onemicron, atol=1e-10):d}].im3"
      actual = r.file
      if expected != actual:
        self.logger.warningglobal(f"rectangle at ({r.cx}, {r.cy}) has the wrong filename {actual}.  Changing it to {expected}.")
      r.file = expected

  @methodtools.lru_cache()
  def getdir(self):
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
  def getoverlaps(self, *, overlaptype=None):
    if overlaptype is None: overlaptype = self.overlaptype
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
  def readallrectangles(self, **extrakwargs):
    rectangles = self.getrectanglelayout()
    return [self.rectangletype(rectangle=r, readingfromfile=False, **self.rectangleextrakwargs, **extrakwargs) for r in rectangles]

class ReadRectanglesIm3FromXML(ReadRectanglesIm3Base, ReadRectanglesFromXML):
  pass

class ReadRectanglesOverlapsFromXML(ReadRectanglesFromXML, ReadRectanglesOverlapsBase):
  def readalloverlaps(self, **kwargs):
    return self.getoverlaps(**kwargs)

class ReadRectanglesOverlapsIm3FromXML(ReadRectanglesOverlapsIm3Base, ReadRectanglesOverlapsFromXML):
  pass

class TempDirSample(SampleBase):
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
