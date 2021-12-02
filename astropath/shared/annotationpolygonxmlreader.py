import abc, argparse, collections, itertools, jxmlease, matplotlib.patches, matplotlib.pyplot as plt, methodtools, more_itertools, numpy as np, pathlib, re
from ..utilities import units
from ..utilities.dataclasses import MetaDataAnnotation, MyDataClassFrozen
from ..utilities.miscmath import floattoint
from ..utilities.tableio import readtable, writetable
from ..utilities.units.dataclasses import distancefield, DataClassWithApscale
from .csvclasses import Annotation, Region, Vertex
from .image_masking.maskloader import MaskLoader
from .logging import dummylogger, printlogger
from .polygon import SimplePolygon
from .qptiff import QPTiff

class AllowedAnnotation(MyDataClassFrozen):
  name: str = MetaDataAnnotation(readfunction=str.lower)
  layer: int
  color: str
  synonyms: set = MetaDataAnnotation(set(), readfunction=lambda x: set(x.lower().split(",")) if x else set(), writefunction=lambda x: ",".join(sorted(x)))

class AnnotationNodeBase(abc.ABC):
  def __init__(self):
    self.usesubindex = None
    self.__newannotationtype = None
  @property
  def usesubindex(self): return self.__usesubindex
  @usesubindex.setter
  def usesubindex(self, value):
    if value is None or value is True or value is False:
      self.__usesubindex = value
    else:
      raise ValueError("usesubindex can only be None, True, or False")
  @property
  @abc.abstractmethod
  def rawname(self): pass
  @property
  def annotationname(self):
    result = self.rawname.lower().strip()
    if self.__newannotationtype is not None:
      result = result.replace(self.__oldannotationtype, self.__newannotationtype)
    if self.usesubindex is None: return result

    regex = " ([0-9]+)$"
    match = re.search(regex, result)
    if self.usesubindex is True:
      if match: return result
      return result + " 1"
    elif self.usesubindex is False:
      if not match: return result
      subindex = match.group(1)
      if subindex == 1:
        return re.sub(regex, "", result)
      raise ValueError(f"Can't force not having a subindex when the subindex is > 1: {result}")

  @property
  def annotationtype(self):
    return re.sub(r" [0-9]+$", "", self.annotationname)
  @annotationtype.setter
  def annotationtype(self, value):
    self.__oldannotationtype = self.annotationtype
    self.__newannotationtype = value
  @property
  def annotationsubindex(self):
    result = self.annotationname.replace(self.annotationtype, "")
    if result:
      return int(self.annotationname.replace(self.annotationtype, ""))
    else:
      return 1

  @property
  @abc.abstractmethod
  def color(self): pass
  @property
  @abc.abstractmethod
  def visible(self): pass
  @property
  @abc.abstractmethod
  def regions(self): pass

class AnnotationNodeXML(AnnotationNodeBase, units.ThingWithApscale):
  def __init__(self, node, *, apscale, **kwargs):
    super().__init__(**kwargs)
    self.__xmlnode = node
    self.__apscale = apscale
  @property
  def apscale(self): return self.__apscale
  @property
  def rawname(self):
    return self.__xmlnode.get_xml_attr("Name")

  @property
  def color(self):
    color = int(self.__xmlnode.get_xml_attr('LineColor'))
    color = f"{color:06X}"
    color = color[4:6] + color[2:4] + color[0:2]
    return color
  @property
  def visible(self):
    return {"true": True, "false": False}[self.__xmlnode.get_xml_attr("Visible").lower().strip()]

  @property
  def regions(self):
    if not self.__xmlnode["Regions"]: return []
    regions = self.__xmlnode["Regions"]["Region"]
    if isinstance(regions, jxmlease.XMLDictNode): regions = regions,
    return [AnnotationRegionXML(_, apscale=self.apscale) for _ in regions]

class AnnotationNodeFromPolygons(AnnotationNodeBase):
  def __init__(self, name, polygons, *, color, visible=True, **kwargs):
    super().__init__(**kwargs)
    self.__name = name
    self.__polygons = polygons
    self.__color = color
    self.__visible = visible
  @property
  def rawname(self):
    return self.__name

  @property
  def color(self):
    return self.__color
  @property
  def visible(self):
    return bool(self.__visible)

  @property
  def regions(self):
    return [AnnotationRegionFromPolygon(p) for p in self.__polygons]

class AnnotationRegionBase(abc.ABC):
  @property
  @abc.abstractmethod
  def vertices(self): pass
  @property
  @abc.abstractmethod
  def NegativeROA(self): pass
  @property
  @abc.abstractmethod
  def Type(self): pass

class AnnotationRegionXML(AnnotationRegionBase, units.ThingWithApscale):
  def __init__(self, xmlnode, *, apscale, **kwargs):
    super().__init__(**kwargs)
    self.__xmlnode = xmlnode
    self.__apscale = apscale
  @property
  def apscale(self): return self.__apscale
  @property
  def vertices(self):
    vertices = self.__xmlnode["Vertices"]["V"]
    if isinstance(vertices, jxmlease.XMLDictNode): vertices = vertices,
    return [AnnotationVertexXML(_, apscale=self.apscale) for _ in vertices]
  @property
  def NegativeROA(self): return bool(int(self.__xmlnode.get_xml_attr("NegativeROA")))
  @property
  def Type(self): return self.__xmlnode.get_xml_attr("Type")

class AnnotationRegionFromPolygon(AnnotationRegionBase):
  def __init__(self, polygon, **kwargs):
    super().__init__(**kwargs)
    self.__polygon = polygon
  @property
  def vertices(self):
    vertices = self.__xmlnode["Vertices"]["V"]
    if isinstance(vertices, jxmlease.XMLDictNode): vertices = vertices,
    return [AnnotationVertexXML(_) for _ in vertices]
  @property
  def NegativeROA(self): return False
  @property
  def Type(self): return "Polygon"

class AnnotationVertexBase(abc.ABC):
  @property
  @abc.abstractmethod
  def X(self): pass
  @property
  @abc.abstractmethod
  def Y(self): pass
class AnnotationVertexXML(AnnotationVertexBase, units.ThingWithApscale):
  def __init__(self, node, *, apscale, **kwargs):
    super().__init__(**kwargs)
    self.__xmlnode = node
    self.__apscale = apscale
  @property
  def apscale(self): return self.__apscale
  @property
  def X(self): return int(self.__xmlnode.get_xml_attr("X")) * self.oneappixel
  @property
  def Y(self): return int(self.__xmlnode.get_xml_attr("Y")) * self.oneappixel

class AnnotationVertexFromPolygon(DataClassWithApscale):
  X: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int, pscalename="apscale")
  Y: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int, pscalename="apscale")

class XMLPolygonAnnotationReader(units.ThingWithPscale, units.ThingWithApscale):
  """
  Class to read the annotations from the annotations.polygons.xml file
  """
  def __init__(self, xmlfile, *, pscale=None, apscale=None, logger=dummylogger, saveallimages=False, imagefolder=None, imagefiletype="pdf", annotationsynonyms=None, reorderannotations=False):
    self.xmlfile = pathlib.Path(xmlfile)
    self.__logger = logger
    self.__saveallimages = saveallimages
    if imagefolder is not None: imagefolder = pathlib.Path(imagefolder)
    self.__imagefolder = imagefolder
    if pscale is None: pscale = 1
    if apscale is None:
      if self.__imagefolder is not None:
        with QPTiff(self.qptifffilename) as fqptiff:
          apscale = fqptiff.apscale
      else:
        apscale = 1
    self.__imagefiletype = imagefiletype
    self.__pscale = pscale
    self.__apscale = apscale
    if annotationsynonyms is None:
      annotationsynonyms = {}
    self.__annotationsynonyms = annotationsynonyms
    self.__reorderannotations = reorderannotations
    self.allowedannotations #make sure there are no duplicate synonyms etc.
  @property
  def pscale(self): return self.__pscale
  @property
  def apscale(self): return self.__apscale
  @property
  def qptifffilename(self):
    return self.xmlfile.with_suffix("").with_suffix("").with_suffix(".qptiff")

  @methodtools.lru_cache()
  @property
  def allowedannotations(self):
    result = readtable(pathlib.Path(__file__).parent/"master_annotation_list.csv", AllowedAnnotation)
    for synonym, name in self.__annotationsynonyms.items():
      synonym = synonym.lower()
      name = name.lower()
      if any(synonym in {a.name} | a.synonyms for a in result):
        raise ValueError("Duplicate synonym: {synonym}")
      try:
        a, = {a for a in result if name in {a.name} | a.synonyms}
      except ValueError:
        raise ValueError(f"Unknown annotation for synonym: {name}")
      a.synonyms.add(synonym)
    return result

  def allowedannotation(self, nameornumber, *, logwarning=True):
    try:
      result, = {a for a in self.allowedannotations if nameornumber in {a.layer, a.name} | a.synonyms}
    except ValueError:
      typ = 'number' if isinstance(nameornumber, int) else 'name'
      raise ValueError(f"Unknown annotation {typ} {nameornumber}")
    if logwarning and nameornumber not in {result.layer, result.name}:
      self.__logger.warningglobal(f"renaming annotation {nameornumber} to {result.name}")
    return result

  @property
  def annotationnodes(self):
    with open(self.xmlfile, "rb") as f:
      return [AnnotationNodeXML(node, apscale=self.apscale) for _, _, node in jxmlease.parse(f, generator="/Annotations/Annotation")]

  def getXMLpolygonannotations(self):
    annotations = []
    allregions = []
    allvertices = []

    errors = []

    nodes = self.annotationnodes

    count = more_itertools.peekable(itertools.count(1))
    for node in nodes[:]:
      if not node.regions:
        if node.annotationtype != "empty":
          self.__logger.warningglobal(f"Annotation {node.annotationname} is empty, skipping it")
        nodes.remove(node)
    for node in nodes:
      try:
        node.annotationtype = self.allowedannotation(node.annotationtype).name
      except ValueError:
        pass

    def annotationorder(node):
      try:
        return self.allowedannotation(node.annotationtype, logwarning=False).layer, node.annotationsubindex
      except ValueError:
        return float("inf"), 0
    nodes.sort(key=annotationorder)

    nodesbytype = collections.defaultdict(lambda: [])
    for node in nodes:
      nodesbytype[node.annotationtype].append(node)
    for node in nodes:
      if len(nodesbytype[node.annotationtype]) > 1:
        node.usesubindex = True
      else:
        node.usesubindex = False

    for layeridx, (annotationtype, annotationnodes) in zip(count, nodesbytype.items()):
      try:
        targetannotation = self.allowedannotation(annotationtype)
      except ValueError as e:
        errors.append(str(e))
        continue
      subindices = [node.annotationsubindex for node in annotationnodes]
      if subindices != list(range(1, len(subindices)+1)):
        errors.append(f"Annotation subindices for {annotationtype} are not sequential: {', '.join(str(subindex) for subindex in subindices)}")
        continue
      annotationtype = targetannotation.name
      targetlayer = targetannotation.layer
      targetcolor = targetannotation.color

      if layeridx > targetlayer:
        assert False
      else:
        while layeridx < targetlayer:
          emptycolor = self.allowedannotation(layeridx).color
          annotations.append(
            Annotation(
              color=emptycolor,
              visible=False,
              name="empty",
              sampleid=0,
              layer=layeridx,
              poly="poly",
              pscale=self.pscale,
              apscale=self.apscale,
            )
          )
          layeridx = next(count)

      for node in annotationnodes:
        color = node.color
        visible = node.visible
        if node.usesubindex:
          name = f"{annotationtype} {node.annotationsubindex}"
          layer = layeridx * 1000 + node.annotationsubindex
        else:
          name = annotationtype
          layer = layeridx
        if color != targetcolor:
          self.__logger.warning(f"Annotation {name} has the wrong color {color}, changing it to {targetcolor}")
          color = targetcolor
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

        regions = node.regions
        if not regions: continue
        m = 1
        for region in regions:
          regioncounter = itertools.count(m)
          regionid = 1000*layer + m
          vertices = region.vertices
          regionvertices = []
          for k, vertex in enumerate(vertices, start=1):
            x = vertex.X
            y = vertex.Y
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
          isNeg = region.NegativeROA

          polygon = SimplePolygon(vertices=regionvertices)
          valid = polygon.makevalid(round=True, imagescale=self.apscale)

          perimeter = 0
          maxlength = 0
          longestidx = None
          for nlines, (v1, v2) in enumerate(more_itertools.pairwise(regionvertices+[regionvertices[0]]), start=1):
            length = np.sum((v1.xvec-v2.xvec)**2)**.5
            if not length: continue
            maxlength, longestidx = max((maxlength, longestidx), (length, nlines))
            perimeter += length

          saveimage = self.__saveallimages
          badimage = False
          if (longestidx == 1 or longestidx == len(regionvertices)) and maxlength / (perimeter/nlines) > 30:
            self.__logger.warningglobal(f"annotation polygon might not be closed: region id {regionid}")
            saveimage = True
            badimage = True

          if saveimage and self.__imagefolder is not None:
            poly = SimplePolygon(vertices=regionvertices)
            with QPTiff(self.qptifffilename) as fqptiff:
              zoomlevel = fqptiff.zoomlevels[0]
              qptiff = zoomlevel[0].asarray()
              pixel = self.oneappixel
              xymin = np.min(poly.vertexarray, axis=0).astype(units.unitdtype)
              xymax = np.max(poly.vertexarray, axis=0).astype(units.unitdtype)
              xybuffer = (xymax - xymin) / 20
              xymin -= xybuffer
              xymax += xybuffer
              (xmin, ymin), (xmax, ymax) = xymin, xymax
              fig, ax = plt.subplots(1, 1)
              plt.imshow(
                qptiff[
                  floattoint(float(ymin//pixel)):floattoint(float(ymax//pixel)),
                  floattoint(float(xmin//pixel)):floattoint(float(xmax//pixel)),
                ],
                extent=[float(xmin//pixel), float(xmax//pixel), float(ymax//pixel), float(ymin//pixel)],
              )
              ax.add_patch(poly.matplotlibpolygon(fill=False, color="red", imagescale=self.apscale))

            if badimage:
              openvertex1 = poly.vertexarray[0]
              openvertex2 = poly.vertexarray[{1: 1, len(regionvertices): -1}[longestidx]]
              boxxmin, boxymin = np.min([openvertex1, openvertex2], axis=0) - xybuffer/2
              boxxmax, boxymax = np.max([openvertex1, openvertex2], axis=0) + xybuffer/2
              ax.add_patch(matplotlib.patches.Rectangle((boxxmin//pixel, boxymin//pixel), (boxxmax-boxxmin)//pixel, (boxymax-boxymin)//pixel, color="violet", fill=False))

            fig.savefig(self.__imagefolder/self.xmlfile.with_suffix("").with_suffix("").with_suffix(f".annotation-{regionid}.{self.__imagefiletype}").name)
            plt.close(fig)

          for subpolygon in valid:
            for polygon, m in zip([subpolygon.outerpolygon] + subpolygon.subtractpolygons, regioncounter): #regioncounter has to be last! https://www.robjwells.com/2019/06/help-zip-is-eating-my-iterators-items/
              regionid = 1000*layer + m
              polygon.regionid = regionid
              regionvertices = polygon.outerpolygon.vertices

              allvertices += list(regionvertices)

              allregions.append(
                Region(
                  regionid=regionid,
                  sampleid=0,
                  layer=layer,
                  rid=m,
                  isNeg=isNeg ^ (polygon is not subpolygon.outerpolygon),
                  type=region.Type,
                  nvert=len(regionvertices),
                  poly=None,
                  apscale=self.apscale,
                  pscale=self.pscale,
                )
              )

          m = next(regioncounter)

    if "good tissue" not in nodesbytype:
      errors.append(f"Didn't find a 'good tissue' annotation (only found: {', '.join(nodesbytype)})")

    if errors:
      raise ValueError("\n".join(errors))

    return annotations, allregions, allvertices

class XMLPolygonAnnotationReaderWithOutline(XMLPolygonAnnotationReader, MaskLoader):
  def __init__(self, *args, maskfilename, **kwargs):
    self.__maskfilename = pathlib.Path(maskfilename)
    super().__init__(*args, **kwargs)

  @property
  def maskfilename(self):
    return self.__maskfilename

  @property
  def annotationnodes(self):
    result = super().annotationnodes
    result.append(AnnotationNodeFromPolygons("outline", self.maskpolygons, color=self.allowedannotation("outline").color))
    return result

def writeannotationcsvs(dbloadfolder, xmlfile, csvprefix=None, **kwargs):
  dbloadfolder = pathlib.Path(dbloadfolder)
  dbloadfolder.mkdir(parents=True, exist_ok=True)
  annotations, regions, vertices = XMLPolygonAnnotationReader(xmlfile, **kwargs).getXMLpolygonannotations()
  if csvprefix is None:
    csvprefix = ""
  elif csvprefix.endswith("_"):
    pass
  else:
    csvprefix = csvprefix+"_"
  writetable(dbloadfolder/f"{csvprefix}annotations.csv", annotations)
  writetable(dbloadfolder/f"{csvprefix}regions.csv", regions)
  writetable(dbloadfolder/f"{csvprefix}vertices.csv", vertices)

class AddToDict(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    k, v = values
    dct = getattr(namespace, self.dest)
    if dct is None: dct = {}; setattr(namespace, self.dest, dct)
    dct[k] = v

def add_rename_annotation_argument(argumentparser):
  argumentparser.add_argument("--rename-annotation", nargs=2, action=AddToDict, dest="annotationsynonyms", metavar=("XMLNAME", "NEWNAME"), help="Rename an annotation given in the xml file to a new name (which has to be in the master list)")
  argumentparser.add_argument("--reorder-annotations", action="store_true", dest="reorderannotations", help="Reorder annotations if they are in the wrong order")

def main(args=None):
  p = argparse.ArgumentParser(description="read an annotations.polygons.xml file and write out csv files for the annotations, regions, and vertices")
  p.add_argument("dbloadfolder", type=pathlib.Path, help="folder to write the output csv files in")
  p.add_argument("xmlfile", type=pathlib.Path, help="path to the annotations.polygons.xml file")
  p.add_argument("--csvprefix", help="prefix to put in front of the csv file names")
  add_rename_annotation_argument(p)
  args = p.parse_args(args=args)
  with units.setup_context("fast"):
    writeannotationcsvs(**args.__dict__, logger=printlogger("annotations"))

def checkannotations(args=None):
  p = argparse.ArgumentParser(description="run astropath checks on an annotations.polygons.xml file")
  p.add_argument("xmlfile", type=pathlib.Path, help="path to the annotations.polygons.xml file")
  g = p.add_mutually_exclusive_group()
  g.add_argument("--save-polygon-images", action="store_const", dest="imagefolder", const=pathlib.Path("."), help="save all annotation images to the current folder")
  g.add_argument("--save-polygon-images-folder", type=pathlib.Path, dest="imagefolder", help="save all annotation images to the given directory")
  g.add_argument("--save-bad-polygon-images", action="store_const", dest="badimagefolder", const=pathlib.Path("."), help="if there are unclosed annotations, save a debug image to the current directory pointing out the problem")
  g.add_argument("--save-bad-polygon-images-folder", type=pathlib.Path, dest="badimagefolder", help="if there are unclosed annotations, save a debug image to the given directory pointing out the problem")
  p.add_argument("--save-images-filetype", default="pdf", choices=("pdf", "png"), dest="imagefiletype", help="image format to save debug images")
  add_rename_annotation_argument(p)
  args = p.parse_args(args=args)
  if args.imagefolder is not None:
    args.saveallimages = True
  else:
    args.imagefolder = args.badimagefolder
  del args.badimagefolder
  logger = printlogger("annotations")
  with units.setup_context("fast"):
    XMLPolygonAnnotationReader(**args.__dict__, logger=logger).getXMLpolygonannotations()
  logger.info(f"{args.xmlfile} looks good!")
