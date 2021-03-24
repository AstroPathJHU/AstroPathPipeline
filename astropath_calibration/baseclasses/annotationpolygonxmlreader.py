import argparse, collections, itertools, jxmlease, more_itertools, pathlib
from ..utilities import units
from ..utilities.tableio import writetable
from .csvclasses import Annotation, Region, Vertex

class XMLPolygonAnnotationReader(units.ThingWithPscale, units.ThingWithApscale):
  def __init__(self, xmlfile, pscale, apscale):
    self.xmlfile = pathlib.Path(xmlfile)
    self.__pscale = pscale
    self.__apscale = apscale
  @property
  def pscale(self): return self.__pscale
  @property
  def apscale(self): return self.__apscale

  def getXMLpolygonannotations(self):
    annotations = []
    allregions = []
    allvertices = []

    AllowedAnnotation = collections.namedtuple("AllowedAnnotation", "name layer color")
    allowedannotations = [
      AllowedAnnotation("good tissue", 1, "FFFF00"),
      AllowedAnnotation("tumor", 2, "00FF00"),
      AllowedAnnotation("lymph node", 3, "FF0000"),
      AllowedAnnotation("regression", 4, "00FFFF"),
      AllowedAnnotation("epithelial islands", 5, "FF00FF"),
    ]
    def allowedannotation(nameornumber):
      try:
        result, = {a for a in allowedannotations if nameornumber in (a.layer, a.name)}
      except ValueError:
        typ = 'number' if isinstance(nameornumber, int) else 'name'
        raise ValueError(f"Unknown annotation {typ} {nameornumber}")
      return result

    with open(self.xmlfile, "rb") as f:
      count = more_itertools.peekable(itertools.count(1))
      for layer, (path, _, node) in zip(count, jxmlease.parse(f, generator="/Annotations/Annotation")):
        color = f"{int(node.get_xml_attr('LineColor')):06X}"
        color = color[4:6] + color[2:4] + color[0:2]
        visible = {
          "true": True,
          "false": False,
        }[node.get_xml_attr("Visible").lower().strip()]
        name = node.get_xml_attr("Name").lower().strip()
        if name == "empty":
          count.prepend(layer)
          continue
        targetannotation = allowedannotation(name)
        targetlayer = targetannotation.layer
        targetcolor = targetannotation.color
        if layer > targetlayer:
          raise ValueError(f"Annotations are in the wrong order: target order is {', '.join(_.name for _ in allowedannotations)}, but {name} is after {annotations[-1].name}")
        if color != targetcolor:
          raise ValueError(f"Annotation {name} has the wrong color {color}, expected {targetcolor}")
        while layer < targetlayer:
          emptycolor = allowedannotation(layer).color
          annotations.append(
            Annotation(
              color=emptycolor,
              visible=False,
              name="empty",
              sampleid=0,
              layer=layer,
              poly="poly",
              pscale=self.pscale,
              apscale=self.apscale,
            )
          )
          layer = next(count)
        annotations.append(
          Annotation(
            color=color,
            visible=visible,
            name=name,
            sampleid=0,
            layer=layer,
            poly="poly",
            pscale=self.pscale,
            apscale=self.apscale,
          )
        )
        if not any(a.name == "good tissue" for a in annotations):
          raise ValueError(f"Didn't find a 'good tissue' annotation (only found: {', '.join(_.name for _ in annotations if _.name != 'empty')})")

        if not node["Regions"]: continue
        regions = node["Regions"]["Region"]
        if isinstance(regions, jxmlease.XMLDictNode): regions = regions,
        for m, region in enumerate(regions, start=1):
          regionid = 1000*layer + m
          vertices = region["Vertices"]["V"]
          if isinstance(vertices, jxmlease.XMLDictNode): vertices = vertices,
          regionvertices = []
          for k, vertex in enumerate(vertices, start=1):
            x = int(vertex.get_xml_attr("X")) * self.oneappixel
            y = int(vertex.get_xml_attr("Y")) * self.oneappixel
            regionvertices.append(
              Vertex(
                regionid=regionid,
                vid=k,
                x=x,
                y=y,
                apscale=self.apscale,
                pscale=self.pscale,
              )
            )
          allvertices += regionvertices

          isNeg = bool(int(region.get_xml_attr("NegativeROA")))
          if isNeg: regionvertices.reverse()
          polygonvertices = []
          for vertex in regionvertices:
            polygonvertices.append(vertex)

          allregions.append(
            Region(
              regionid=regionid,
              sampleid=0,
              layer=layer,
              rid=m,
              isNeg=isNeg,
              type=region.get_xml_attr("Type"),
              nvert=len(vertices),
              poly=None,
              apscale=self.apscale,
              pscale=self.pscale,
            )
          )

    return annotations, allregions, allvertices

def writeannotationcsvs(dbloadfolder, xmlfile, pscale, apscale, csvprefix=None):
  dbloadfolder = pathlib.Path(dbloadfolder)
  dbloadfolder.mkdir(parents=True, exist_ok=True)
  annotations, regions, vertices = XMLPolygonAnnotationReader(xmlfile, pscale, apscale).getXMLpolygonannotations()
  if csvprefix is None:
    csvprefix = ""
  elif csvprefix.endswith("_"):
    pass
  else:
    csvprefix = csvprefix+"_"
  writetable(dbloadfolder/f"{csvprefix}annotations.csv", annotations)
  writetable(dbloadfolder/f"{csvprefix}regions.csv", regions)
  writetable(dbloadfolder/f"{csvprefix}vertices.csv", vertices)

def main(args=None):
  p = argparse.ArgumentParser()
  p.add_argument("dbloadfolder", type=pathlib.Path)
  p.add_argument("xmlfile", type=pathlib.Path)
  p.add_argument("pscale", type=float)
  p.add_argument("apscale", type=float)
  p.add_argument("--csvprefix")
  args = p.parse_args(args=args)
  with units.setup_context("fast"):
    writeannotationcsvs(**args.__dict__)
