import abc, dataclasses, datetime, dateutil, jxmlease, methodtools, numpy as np, pathlib
from ..alignment.rectangle import Rectangle
from ..utilities import units
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

class AnnotationXMLReader:
  def __init__(self, filename, *, pscale):
    self.__filename = filename
    self.__pscale = pscale

  @property
  def pscale(self): return self.__pscale

  @methodtools.lru_cache()
  def getdata(self):
    rectangles = []
    globals = []
    perimeters = []
    maxdepth = 1
    with open(self.__filename, "rb") as f:
      for path, _, node in jxmlease.parse(
        f,
        generator="/AnnotationList/Annotations/Annotations-i"
      ):
        annotation = AnnotationFactory(node, pscale=self.pscale)
        globalkwargs = annotation.globals
        if globalkwargs is not None: globals.append(Globals(n=len(globals)+1, **globalkwargs))
        perimeterkwargs = annotation.perimeters
        if perimeterkwargs is not None: perimeters += [
          Perimeter(m=len(perimeters)+1, **kwargs)
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
              file=pathlib.Path(field.im3path.replace("\\", "/")).name,
              pscale=self.pscale,
              readingfromfile=False,
            )
          )

    return rectangles, globals, perimeters

  @property
  def rectangles(self): return self.getdata()[0]
  @property
  def globals(self): return self.getdata()[1]
  @property
  def perimeters(self): return self.getdata()[2]

class AnnotationBase(abc.ABC):
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
  @abc.abstractproperty
  def fields(self): pass
  @abc.abstractproperty
  def globals(self): pass
  @abc.abstractproperty
  def perimeters(self): pass
  @property
  def subtype(self): return self.__xmlnode.get_xml_attr("subtype")

class RectangleAnnotation(AnnotationBase):
  @property
  def fields(self): return self,
  @property
  def history(self):
    history = self.xmlnode["History"]["History-i"]
    if isinstance(history, jxmlease.XMLDictNode): history = history,
    return history
  @property
  def isacquired(self):
    return self.history[-1]["Type"] == "Acquired"
  @property
  def im3path(self):
    return self.history[-1]["Im3Path"]
  @property
  def x(self): return units.Distance(microns=float(self.xmlnode["Bounds"]["Origin"]["X"]), pscale=self.pscale)
  @property
  def y(self): return units.Distance(microns=float(self.xmlnode["Bounds"]["Origin"]["Y"]), pscale=self.pscale)
  @property
  def w(self): return units.Distance(microns=float(self.xmlnode["Bounds"]["Size"]["Width"]), pscale=self.pscale)
  @property
  def h(self): return units.Distance(microns=float(self.xmlnode["Bounds"]["Size"]["Height"]), pscale=self.pscale)
  @property
  def cx(self):
    return units.Distance(microns=int(np.round(units.microns(self.x+0.5*self.w))), pscale=self.pscale)
  @property
  def cy(self):
    return units.Distance(microns=int(np.round(units.microns(self.y+0.5*self.h))), pscale=self.pscale)
  @property
  def time(self): return dateutil.parser.parse(self.history[-1]["TimeStamp"])

  @property
  def globals(self): return None
  @property
  def perimeters(self): return None

class ROIAnnotation(AnnotationBase):
  @property
  def fields(self):
    fields = self.xmlnode["Fields"]["Fields-i"]
    if isinstance(fields, jxmlease.XMLDictNode): fields = fields,
    for field in fields:
      yield from AnnotationFactory(field, nestdepth=self.nestdepth+1, pscale=self.pscale).fields
  @property
  def globals(self):
    return {
      "x": units.Distance(microns=self.xmlnode["Bounds"]["Origin"]["X"]),
      "y": units.Distance(microns=self.xmlnode["Bounds"]["Origin"]["Y"]),
      "Width": units.Distance(microns=self.xmlnode["Bounds"]["Size"]["Width"]),
      "Height": units.Distance(microns=self.xmlnode["Bounds"]["Size"]["Height"]),
      "Unit": "microns",
      "Tc": self.xmlnode["History"]["History-i"]["TimeStamp"],
    }
  @property
  def perimeters(self):
    result = []
    perimeters = self.xmlnode["Perimeter"]["Perimeter-i"]
    if isinstance(perimeters, jxmlease.XMLDictNode): perimeters = perimeters,
    for i, perimeter in enumerate(perimeters, start=1):
      result.append({
        "n": i,
        "x": units.Distance(microns=perimeter["X"]),
        "y": units.Distance(microns=perimeter["Y"]),
      })
    return result

def AnnotationFactory(xmlnode, **kwargs):
  return {
    "RectangleAnnotation": RectangleAnnotation,
    "ROIAnnotation": ROIAnnotation,
  }[xmlnode.get_xml_attr("subtype")](xmlnode, **kwargs)

@dataclasses.dataclass
class Globals(DataClassWithDistances):
  pixelsormicrons = "microns"

  x: distancefield(pixelsormicrons=pixelsormicrons)
  y: distancefield(pixelsormicrons=pixelsormicrons)
  Width: distancefield(pixelsormicrons=pixelsormicrons)
  Height: distancefield(pixelsormicrons=pixelsormicrons)
  Unit: str
  Tc: datetime.datetime = dataclasses.field(metadata={"writefunction": datetime.datetime.timestamp})

@dataclasses.dataclass
class Perimeter(DataClassWithDistances):
  pixelsormicrons = "microns"

  n: int
  x: distancefield(pixelsormicrons=pixelsormicrons)
  y: distancefield(pixelsormicrons=pixelsormicrons)
