import abc, dateutil, jxmlease, methodtools, numpy as np, pathlib
from ..baseclasses.csvclasses import Globals, Perimeter
from ..baseclasses.rectangle import Rectangle
from ..utilities import units
from ..utilities.misc import floattoint

class AnnotationXMLReader(units.ThingWithPscale):
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
    microscopename = None
    with open(self.__filename, "rb") as f:
      for path, _, node in jxmlease.parse(
        f,
        generator="/AnnotationList/Annotations/Annotations-i"
      ):
        annotation = AnnotationFactory(node, pscale=self.pscale)
        globalkwargs = annotation.globals
        if globalkwargs is not None: globals.append(Globals(**globalkwargs, pscale=self.pscale))
        perimeterkwargs = annotation.perimeters
        if perimeterkwargs is not None: perimeters += [
          Perimeter(**kwargs, pscale=self.pscale)
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
  def microscopename(self): pass
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
  def microscopename(self):
    if not self.isacquired: return None
    return self.history[-1]["UserName"]
  @property
  def x(self): return float(self.xmlnode["Bounds"]["Origin"]["X"]) * self.onemicron
  @property
  def y(self): return float(self.xmlnode["Bounds"]["Origin"]["Y"]) * self.onemicron
  @property
  def w(self): return float(self.xmlnode["Bounds"]["Size"]["Width"]) * self.onemicron
  @property
  def h(self): return float(self.xmlnode["Bounds"]["Size"]["Height"]) * self.onemicron
  @property
  def cx(self):
    return floattoint(np.round(float((self.x+0.5*self.w) / self.onemicron))) * self.onemicron
  @property
  def cy(self):
    return floattoint(np.round(float((self.y+0.5*self.h) / self.onemicron))) * self.onemicron
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
      "x": float(self.xmlnode["Bounds"]["Origin"]["X"]) * self.onemicron,
      "y": float(self.xmlnode["Bounds"]["Origin"]["Y"]) * self.onemicron,
      "Width": float(self.xmlnode["Bounds"]["Size"]["Width"]) * self.onemicron,
      "Height": float(self.xmlnode["Bounds"]["Size"]["Height"]) * self.onemicron,
      "Unit": "microns",
      "Tc": dateutil.parser.parse(self.xmlnode["History"]["History-i"]["TimeStamp"]),
    }
  @property
  def perimeters(self):
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
    names = {field.microscopename for field in self.fields if field.microscopename is not None}
    if not names:
      return None
    if len(names) > 1:
      raise ValueError("Multiple microscope names: "+", ".join(names))
    return names.pop()

def AnnotationFactory(xmlnode, **kwargs):
  return {
    "RectangleAnnotation": RectangleAnnotation,
    "ROIAnnotation": ROIAnnotation,
  }[xmlnode.get_xml_attr("subtype")](xmlnode, **kwargs)
