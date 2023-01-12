import abc, dateutil, jxmlease, methodtools, numpy as np, pathlib
from ..utilities import units
from ..utilities.miscmath import floattoint
from .csvclasses import ROIGlobals, ROIPerimeter
from .rectangle import Rectangle, TMARectangle

class AnnotationXMLReader(units.ThingWithPscale):
  """
  Class to read the annotations from an xml file
  """
  def __init__(self, filename, *, pscale, logger, includehpfsflaggedforacquisition=True, xmlfolder=None, SlideID=None):
    self.__filename = filename
    self.__pscale = pscale
    self.__logger = logger
    self.__xmlfolder = xmlfolder
    self.__includehpfsflaggedforacquisition = includehpfsflaggedforacquisition
    self.__SlideID = SlideID

  @property
  def pscale(self): return self.__pscale

  @methodtools.lru_cache()
  def getdata(self):
    """
    Reads the annotations and gives the rectangles,
    global variables, perimeters, and microscope name
    """
    rectangles = []
    globals = []
    perimeters = []
    maxdepth = 1
    microscopename = None
    with open(self.__filename, "rb") as f:
      for path, _, node in jxmlease.parse(
        f,
        generator="/AnnotationList/Annotations/Annotations-i"
      ):
        annotation = AnnotationFactory(node, pscale=self.pscale)
        globalkwargs = annotation.globals
        if globalkwargs is not None: globals.append(ROIGlobals(**globalkwargs, pscale=self.pscale))
        perimeterkwargs = annotation.perimeters
        if perimeterkwargs is not None: perimeters += [
          ROIPerimeter(**kwargs, pscale=self.pscale)
            for kwargs in perimeterkwargs
        ]

        for field in annotation.fields:
          if field.nestdepth > 2:
            raise ValueError("Found an ROIAnnotation within another ROIAnnotation, did not expect this")
          #don't use RectangleAnnotations if there are also ROIAnnotations
          if field.nestdepth != maxdepth:
            self.__logger.warningglobalonenter("There are manually added HPFs")
            if field.nestdepth < maxdepth:
              continue
            elif field.nestdepth > maxdepth:
              del rectangles[:]
              maxdepth = field.nestdepth
            else:
              assert False

          if not (
            field.isacquired
            or field.isflaggedforacquisition and self.__includehpfsflaggedforacquisition
          ): continue
          rectangles.append(
            field.rectangletype(
              n=len(rectangles)+1,
              **field.rectanglekwargs,
              pscale=self.pscale,
              readingfromfile=False,
              xmlfolder=self.__xmlfolder,
              SlideID=self.__SlideID,
            )
          )
          if microscopename is None is not annotation.microscopename:
            microscopename = str(annotation.microscopename)
          elif microscopename != annotation.microscopename:
            raise ValueError(f"Found multiple different microscope names '{microscopename}' '{annotation.microscopename}'")

    return rectangles, globals, perimeters, microscopename

  @property
  def rectangles(self): return self.getdata()[0]
  @property
  def globals(self): return self.getdata()[1]
  @property
  def perimeters(self): return self.getdata()[2]
  @property
  def microscopename(self): return self.getdata()[3]

class AnnotationBase(units.ThingWithPscale):
  """
  There are two kinds of annotations in the xml files,
  depending on the microscope model and software version.
  This is the base class for both of them. 
  """
  def __init__(self, xmlnode, *, pscale, nestdepth=1):
    self.__xmlnode = xmlnode
    self.__nestdepth = nestdepth
    self.__pscale = pscale
  @property
  def pscale(self): return self.__pscale
  @property
  def xmlnode(self): return self.__xmlnode
  @property
  def nestdepth(self): return self.__nestdepth
  def __str__(self): return str(self.xmlnode)
  @property
  @abc.abstractmethod
  def fields(self):
    """
    RectangleAnnotations for all HPFs within this annotation
    """
  @property
  @abc.abstractmethod
  def globals(self):
    """
    Global variables from the annotation
    """
  @property
  @abc.abstractmethod
  def perimeters(self):
    """
    ROI perimeters from the annotation
    """
  @property
  @abc.abstractmethod
  def microscopename(self):
    """
    The microscope name from the annotation
    (really the name of the computer connected to the microscope)
    """

  @property
  def subtype(self):
    """
    The annotation subtype - RectangleAnnotation or ROIAnnotation
    """
    return self.__xmlnode.get_xml_attr("subtype")

class SimpleAnnotation(AnnotationBase):
  """
  A simple annotation is for a single HPF.
  """
  @property
  def fields(self):
    """
    A simple annotation is for a single HPF
    """
    return self,
  @property
  def history(self):
    """
    The history of the HPF scanning
    """
    history = self.xmlnode["History"]["History-i"]
    if isinstance(history, jxmlease.XMLDictNode): history = history,
    return history
  @property
  def isacquired(self):
    """
    Was this HPF acquired? (i.e. did it finish successfully?)
    """
    return self.acquisitionnode is not None
  @property
  def isflaggedforacquisition(self):
    """
    Was this HPF flagged for acquisition?
    """
    return self.flaggedforacquisitionnode is not None

  @property
  def acquisitionnode(self):
    result = None
    for node in self.history:
      if node["Type"] == "Acquired":
        result = node
        ignored = False
      elif result is not None:
        if node["Type"] == "Ignored":
          ignored = True
        elif node["Type"] == "Unignored":
          ignored = False
        else:
          raise ValueError(f"Unknown history item {node['Type']} after Acquired for {result['Im3Path']}")

    if result is None:
      return None
    if ignored:
      return None
    return result
    if self.isacquired:
      return self.history[-1]

  @property
  def flaggedforacquisitionnode(self):
    result = None
    for node in self.history:
      if node["Type"] == "FlaggedForAcquisition":
        result = node
        ignored = False
        deleted = False
        failed = False
      elif result is not None:
        if node["Type"] == "Acquired":
          pass
        elif node["Type"] == "Ignored":
          ignored = True
        elif node["Type"] == "Unignored":
          ignored = False
        elif node["Type"] == "Deleted":
          deleted = True
        elif node["Type"] == "AcquisitionFailed":
          failed = True
        else:
          raise ValueError(f"Unknown history item {node['Type']} after FlaggedForAcquisition for {result['Im3Path']}")

    if result is None:
      return None
    if ignored or deleted or failed:
      return None
    return result

  @property
  def im3path(self):
    """
    Path to the im3 file
    """
    if not self.isacquired: return None
    return pathlib.PureWindowsPath(self.acquisitionnode["Im3Path"])
  @property
  def microscopename(self):
    """
    Name of the computer operating the microscope
    """
    if not self.isacquired: return None
    return self.acquisitionnode["UserName"]
  @property
  def x(self):
    """
    x position of the HPF
    """
    return float(self.xmlnode["Bounds"]["Origin"]["X"]) * self.onemicron
  @property
  def y(self):
    """
    y position of the HPF
    """
    return float(self.xmlnode["Bounds"]["Origin"]["Y"]) * self.onemicron
  @property
  def w(self):
    """
    width of the HPF
    """
    return float(self.xmlnode["Bounds"]["Size"]["Width"]) * self.onemicron
  @property
  def h(self):
    """
    height of the HPF
    """
    return float(self.xmlnode["Bounds"]["Size"]["Height"]) * self.onemicron
  @property
  def cx(self):
    """
    x of the HPF's center in integer pixels
    """
    return floattoint(np.round(float((self.x+0.5*self.w) / self.onemicron))) * self.onemicron
  @property
  def cy(self):
    """
    y of the HPF's center in integer pixels
    """
    return floattoint(np.round(float((self.y+0.5*self.h) / self.onemicron))) * self.onemicron
  @property
  def time(self):
    """
    time stamp when the HPF was acquired
    """
    if self.isacquired:
      node = self.acquisitionnode
    elif self.isflaggedforacquisition:
      node = self.flaggedforacquisitionnode
    else:
      return None
    return dateutil.parser.parse(node["TimeStamp"])

  @property
  def globals(self): "{type(self).__name__}s don't have global variables"
  @property
  def perimeters(self): "{type(self).__name__}s don't have perimeters"

  @abc.abstractmethod
  def rectangletype(self): pass
  @property
  def rectanglekwargs(self):
    return {
      "x": self.x,
      "y": self.y,
      "cx": self.cx,
      "cy": self.cy,
      "w": self.w,
      "h": self.h,
      "t": self.time,
      "file": self.im3path.relative_to(self.im3path.parent) if self.im3path is not None else None,
    }

class RectangleAnnotation(SimpleAnnotation):
  """
  A RectangleAnnotation is for a single HPF of a normal slide
  """
  @property
  def rectangletype(self): return Rectangle

class TMACoreAnnotation(SimpleAnnotation):
  """
  A TMACoreAnnotation is for a single TMA core
  """
  @property
  def rectangletype(self): return TMARectangle
  @property
  def TMAsector(self):
    return int(self.xmlnode["Sector"])
  @property
  def TMAname(self):
    name1, name2 = (int(_) for _ in self.xmlnode["Name"].split(","))
    return name1, name2
  @property
  def rectanglekwargs(self):
    return {
      **super().rectanglekwargs,
      "TMAsector": self.TMAsector,
      "TMAname1": self.TMAname[0],
      "TMAname2": self.TMAname[1],
    }

class CompoundAnnotation(AnnotationBase):
  """
  An CompoundAnnotation includes multiple HPFs within the region of interest
  """
  @property
  @abc.abstractmethod
  def fieldtagname(self):
    pass

  @property
  def fields(self):
    """
    The HPFs in this ROI
    """
    fields = self.xmlnode[self.fieldtagname][f"{self.fieldtagname}-i"]
    if isinstance(fields, jxmlease.XMLDictNode): fields = fields,
    for field in fields:
      yield from AnnotationFactory(field, nestdepth=self.nestdepth+1, pscale=self.pscale).fields
  @property
  def globals(self):
    """
    Global variables for the ROI
    """
    return {
      "x": float(self.xmlnode["Bounds"]["Origin"]["X"]) * self.onemicron,
      "y": float(self.xmlnode["Bounds"]["Origin"]["Y"]) * self.onemicron,
      "Width": float(self.xmlnode["Bounds"]["Size"]["Width"]) * self.onemicron,
      "Height": float(self.xmlnode["Bounds"]["Size"]["Height"]) * self.onemicron,
      "Unit": "microns",
      "Tc": dateutil.parser.parse(self.xmlnode["History"]["History-i"]["TimeStamp"]),
    }

  @property
  def microscopename(self):
    """
    Name of the computer operating the microscope
    """
    names = {field.microscopename for field in self.fields if field.microscopename is not None}
    if not names:
      return None
    if len(names) > 1:
      raise ValueError("Multiple microscope names: "+", ".join(names))
    return names.pop()

class ROIAnnotation(CompoundAnnotation):
  """
  A ROIAnnotation is for a region of a normal slide
  """
  @property
  def fieldtagname(self):
    return "Fields"
  @property
  def perimeters(self):
    """
    Describes the perimeter of the ROI
    """
    result = []
    perimeters = self.xmlnode["Perimeter"]["Perimeter-i"]
    if isinstance(perimeters, jxmlease.XMLDictNode): perimeters = perimeters,
    for i, perimeter in enumerate(perimeters, start=1):
      result.append({
        "n": i,
        "x": float(perimeter["X"]) * self.onemicron,
        "y": float(perimeter["Y"]) * self.onemicron,
      })
    return result

class TMASectorAnnotation(CompoundAnnotation):
  """
  A TMASectorAnnotation is for a TMA sector
  """
  @property
  def fieldtagname(self):
    return "Cores"
  @property
  def perimeters(self): "{type(self).__name__}s don't have perimeters"

def AnnotationFactory(xmlnode, **kwargs):
  """
  Returns the right kind of annotation, either Rectangle or ROI
  """
  return {
    "RectangleAnnotation": RectangleAnnotation,
    "ROIAnnotation": ROIAnnotation,
    "TMACoreAnnotation": TMACoreAnnotation,
    "TMASectorAnnotation": TMASectorAnnotation,
  }[xmlnode.get_xml_attr("subtype")](xmlnode, **kwargs)
