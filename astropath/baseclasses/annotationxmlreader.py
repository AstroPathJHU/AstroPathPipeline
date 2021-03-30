import abc, dateutil, jxmlease, methodtools, numpy as np, pathlib
from ..baseclasses.csvclasses import ROIGlobals, ROIPerimeter
from ..baseclasses.rectangle import Rectangle
from ..utilities import units
from ..utilities.misc import floattoint

class AnnotationXMLReader(units.ThingWithPscale):
  """
  Class to read the annotations from an xml file
  """
  def __init__(self, filename, *, pscale, xmlfolder=None):
    self.__filename = filename
    self.__pscale = pscale
    self.__xmlfolder = xmlfolder

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
          #(inherited from matlab code, not sure where this logic comes in)
          if field.nestdepth < maxdepth: continue
          if field.nestdepth > maxdepth:
            del rectangles[:]
            maxdepth = field.nestdepth

          if not field.isacquired: continue
          rectangles.append(
            Rectangle(
              n=len(rectangles)+1,
              x=field.x,
              y=field.y,
              cx=field.cx,
              cy=field.cy,
              w=field.w,
              h=field.h,
              t=field.time,
              file=field.im3path.name,
              pscale=self.pscale,
              readingfromfile=False,
              xmlfolder=self.__xmlfolder,
            )
          )
          if microscopename is None:
            microscopename = str(annotation.microscopename)
          elif microscopename != annotation.microscopename:
            raise ValueError("Found multiple different microscope names '{microscopename}' '{annotation.microscopename}'")

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

class RectangleAnnotation(AnnotationBase):
  """
  A rectangle annotation is for a single HPF.
  """
  @property
  def fields(self):
    """
    A rectangle annotation is for a single HPF
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
    return self.history[-1]["Type"] == "Acquired"
  @property
  def im3path(self):
    """
    Path to the im3 file
    """
    return pathlib.PureWindowsPath(self.history[-1]["Im3Path"])
  @property
  def microscopename(self):
    """
    Name of the computer operating the microscope
    """
    if not self.isacquired: return None
    return self.history[-1]["UserName"]
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
    return dateutil.parser.parse(self.history[-1]["TimeStamp"])

  @property
  def globals(self): "RectangleAnnotations don't have global variables"
  @property
  def perimeters(self): "RectangleAnnotations don't have perimeters"

class ROIAnnotation(AnnotationBase):
  """
  An ROIAnnotation includes multiple HPFs within the region of interest
  """
  @property
  def fields(self):
    """
    The HPFs in this ROI
    """
    fields = self.xmlnode["Fields"]["Fields-i"]
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

def AnnotationFactory(xmlnode, **kwargs):
  """
  Returns the right kind of annotation, either Rectangle or ROI
  """
  return {
    "RectangleAnnotation": RectangleAnnotation,
    "ROIAnnotation": ROIAnnotation,
  }[xmlnode.get_xml_attr("subtype")](xmlnode, **kwargs)
