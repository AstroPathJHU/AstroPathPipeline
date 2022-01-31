import abc, argparse, collections, hashlib, itertools, jxmlease, matplotlib.patches, matplotlib.pyplot as plt, methodtools, more_itertools, numpy as np, pathlib, re
from ..utilities import units
from ..utilities.dataclasses import MetaDataAnnotation, MyDataClassFrozen
from ..utilities.misc import ArgParseAddToDict
from ..utilities.miscmath import floattoint
from ..utilities.tableio import boolasintfield, readtable, writetable
from ..utilities.units.dataclasses import distancefield, DataClassWithAnnoscale
from .csvclasses import Annotation, AnnotationInfo, Region, Vertex
from .image_masking.maskloader import TissueMaskLoaderWithPolygons
from .logging import dummylogger, printlogger, ThingWithLogger
from .polygon import SimplePolygon
from .qptiff import QPTiff

class AllowedAnnotation(MyDataClassFrozen):
  name: str = MetaDataAnnotation(readfunction=str.lower)
  layer: int
  color: str
  synonyms: set = MetaDataAnnotation(set(), readfunction=lambda x: set(x.lower().split(",")) if x else set(), writefunction=lambda x: ",".join(sorted(x)))
  isfromxml: bool = boolasintfield()

  @methodtools.lru_cache()
  @classmethod
  def allowedannotations(cls):
    return readtable(pathlib.Path(__file__).parent/"master_annotation_list.csv", cls)

  @methodtools.lru_cache()
  @classmethod
  def allowedannotation(cls, name):
    annotations = {_ for _ in AllowedAnnotation.allowedannotations() if _.name == name}
    try:
      a, = annotations
    except ValueError:
      if len(annotations) > 1:
        assert False, annotations
      raise ValueError(f"Didn't find an annotation with name {name} in master_annotation_list.csv")
    return a

class AnnotationNodeBase(units.ThingWithAnnoscale):
  def __init__(self, *args, annoscale, **kwargs):
    super().__init__(*args, **kwargs)
    self.usesubindex = None
    self.__newannotationtype = None
    self.__annoscale = annoscale
  @property
  def annoscale(self):
    return self.__annoscale
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

  @property
  @abc.abstractmethod
  def isfromxml(self): pass

  @property
  @abc.abstractmethod
  def areacutoff(self): pass

class AnnotationNodeXML(AnnotationNodeBase):
  def __init__(self, node, **kwargs):
    super().__init__(**kwargs)
    self.__xmlnode = node
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
    return [AnnotationRegionXML(_, annoscale=self.annoscale) for _ in regions]

  @property
  def isfromxml(self): return True

  @property
  def areacutoff(self):
    return None #when it's manually drawn, want to keep whatever there is

class AnnotationNodeFromPolygons(AnnotationNodeBase, units.ThingWithAnnoscale):
  def __init__(self, name, polygons, *, color, areacutoff, visible=True, **kwargs):
    super().__init__(**kwargs)
    self.__name = name
    self.__polygons = sorted(polygons, key=lambda x: tuple(x.outerpolygon.vertexarray[0]))
    self.__color = color
    self.__visible = visible
    self.__areacutoff = areacutoff
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
    result = []
    for p in self.__polygons:
      result += [
        AnnotationRegionFromPolygon(p.outerpolygon, annoscale=self.annoscale),
        *(AnnotationRegionFromPolygon(pp, annoscale=self.annoscale, isNeg=1) for pp in p.subtractpolygons),
      ]
    return result

  @property
  def isfromxml(self): return False

  @property
  def areacutoff(self):
    return self.__areacutoff

class AnnotationRegionBase(units.ThingWithAnnoscale):
  def __init__(self, *args, annoscale, **kwargs):
    super().__init__(*args, **kwargs)
    self.__annoscale = annoscale
  @property
  def annoscale(self):
    return self.__annoscale
  @property
  @abc.abstractmethod
  def vertices(self): pass
  @property
  @abc.abstractmethod
  def NegativeROA(self): pass
  @property
  @abc.abstractmethod
  def Type(self): pass

class AnnotationRegionXML(AnnotationRegionBase):
  def __init__(self, xmlnode, **kwargs):
    super().__init__(**kwargs)
    self.__xmlnode = xmlnode
  @property
  def vertices(self):
    vertices = self.__xmlnode["Vertices"]["V"]
    if isinstance(vertices, jxmlease.XMLDictNode): vertices = vertices,
    return [AnnotationVertexXML(_, annoscale=self.annoscale) for _ in vertices]
  @property
  def NegativeROA(self): return bool(int(self.__xmlnode.get_xml_attr("NegativeROA")))
  @property
  def Type(self): return self.__xmlnode.get_xml_attr("Type")

class AnnotationRegionFromPolygon(AnnotationRegionBase, units.ThingWithAnnoscale):
  def __init__(self, polygon, *, isNeg=False, **kwargs):
    super().__init__(**kwargs)
    self.__polygon = polygon.round(imagescale=self.annoscale)
    self.__isNeg = isNeg
  @property
  def vertices(self):
    assert not self.__polygon.subtractpolygons
    return [AnnotationVertexFromPolygon(X=x, Y=y, annoscale=self.annoscale) for x, y in self.__polygon.outerpolygon.vertexarray]
  @property
  def NegativeROA(self): return self.__isNeg
  @property
  def Type(self): return "Polygon"

class AnnotationVertexBase(units.ThingWithAnnoscale):
  @property
  @abc.abstractmethod
  def X(self): pass
  @property
  @abc.abstractmethod
  def Y(self): pass
class AnnotationVertexXML(AnnotationVertexBase):
  def __init__(self, node, *, annoscale, **kwargs):
    super().__init__(**kwargs)
    self.__xmlnode = node
    self.__annoscale = annoscale
  @property
  def annoscale(self): return self.__annoscale
  @property
  def X(self): return int(self.__xmlnode.get_xml_attr("X")) * self.oneannopixel
  @property
  def Y(self): return int(self.__xmlnode.get_xml_attr("Y")) * self.oneannopixel

class AnnotationVertexFromPolygon(DataClassWithAnnoscale):
  X: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="annoscale")
  Y: units.Distance = distancefield(pixelsormicrons="pixels", pscalename="annoscale")

class ThingWithAnnotationInfos(units.ThingWithPscale, units.ThingWithApscale):
  @property
  @abc.abstractmethod
  def annotationinfofile(self): pass
  @property
  @abc.abstractmethod
  def scanfolder(self): pass
  @methodtools.lru_cache()
  def readannotationinfo(self):
    if not self.annotationinfofile.exists():
      raise FileNotFoundError(f"Can't read the annotation info from {self.annotationinfofile} because it doesn't exist")
    return self.readtable(self.annotationinfofile, AnnotationInfo)
  @methodtools.lru_cache()
  @property
  def annotationinfo(self):
    result = self.readannotationinfo()
    hashdict = {}
    for info in result:
      if info.isfromxml:
        if info.xmlpath not in hashdict:
          with open(info.xmlpath, "rb") as f:
            hash = hashlib.sha256()
            hash.update(f.read())
            hashdict[info.xmlpath] = hash.hexdigest()
        if info.xmlsha != hashdict[info.xmlpath]:
          raise ValueError(f"xml hash {info.xmlsha} in the annotation info doesn't match the current hash of {info.xmlpath}")
    return result

  def readtable(self, filename, rowclass, *, extrakwargs=None, **kwargs):
    if extrakwargs is None: extrakwargs = {}
    if issubclass(rowclass, AnnotationInfo):
      extrakwargs["scanfolder"] = self.scanfolder
    return super().readtable(filename=filename, rowclass=rowclass, extrakwargs=extrakwargs, **kwargs)

class XMLPolygonAnnotationFileBase(ThingWithAnnotationInfos):
  @property
  @abc.abstractmethod
  def annotationspolygonsxmlfile(self): pass
  @property
  def qptifffilename(self):
    return self.annotationspolygonsxmlfile.with_suffix("").with_suffix("").with_suffix(".qptiff")
  @property
  def annotationinfofile(self):
    return self.annotationspolygonsxmlfile.with_suffix(".annotationinfo.csv")
  @property
  def annotationinfo(self):
    infos = super().annotationinfo
    myxmlfile = self.annotationspolygonsxmlfile
    for info in infos:
      infoxmlfile = info.xmlpath
      if infoxmlfile != myxmlfile:
        raise ValueError(f"Expected xmlfile = {myxmlfile}, found {infoxmlfile}")
    return infos

class XMLPolygonAnnotationFile(XMLPolygonAnnotationFileBase):
  def __init__(self, *args, xmlfile, pscale, apscale, **kwargs):
    super().__init__(*args, **kwargs)
    self.__xmlfile = xmlfile
    self.__pscale = pscale
    self.__apscale = apscale
  @property
  def pscale(self): return self.__pscale
  @property
  def apscale(self): return self.__apscale
  @property
  def annotationspolygonsxmlfile(self): return self.__xmlfile
  @property
  def scanfolder(self): return self.annotationspolygonsxmlfile.parent

class MergedAnnotationFiles(ThingWithAnnotationInfos):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  @methodtools.lru_cache()
  @property
  def __xmldict(self):
    xmldict = {}
    for info in self.annotationinfo:
      if info.isfromxml:
        if info.xmlpath not in xmldict:
          with open(info.xmlpath, "rb") as f:
            xmldict[info.xmlpath] = {node.get_xml_attr("Name").lower(): AnnotationNodeXML(node, annoscale=info.annoscale) for _, _, node in jxmlease.parse(f, generator="/Annotations/Annotation")}
    return xmldict
  def getannotationnode(self, info):
    if info.isfromxml:
      return self.__xmldict[info.xmlpath][info.originalname.lower()]
    else:
      raise ValueError(f"Don't know how to get the node for {info}")
  @methodtools.lru_cache()
  @property
  def annotationnodes(self):
    return [self.getannotationnode(info) for info in self.annotationinfo]

class XMLPolygonAnnotationReader(MergedAnnotationFiles, units.ThingWithApscale, ThingWithLogger):
  """
  Class to read the annotations from the annotations.polygons.xml file
  """
  def __init__(self, *args, saveallannotationimages=False, annotationimagefolder=None, annotationimagefiletype="pdf", annotationsynonyms=None, **kwargs):
    self.__saveallannotationimages = saveallannotationimages
    if annotationimagefolder is not None: annotationimagefolder = pathlib.Path(annotationimagefolder)
    self.__annotationimagefolder = annotationimagefolder
    self.__annotationimagefiletype = annotationimagefiletype
    if annotationsynonyms is None:
      annotationsynonyms = {}
    self.__annotationsynonyms = annotationsynonyms
    self.allowedannotations #make sure there are no duplicate synonyms etc.
    super().__init__(*args, **kwargs)

  @property
  def annotationimagefolder(self): return self.__annotationimagefolder
  @property
  @abc.abstractmethod
  def SampleID(self): pass

  @methodtools.lru_cache()
  @property
  def allowedannotations(self):
    result = AllowedAnnotation.allowedannotations()
    allsynonyms = {synonym.lower() for synonym in self.__annotationsynonyms}
    for a in result:
      if a.name.lower() not in allsynonyms:
        a.synonyms.add(a.name.lower())

    for synonym, name in self.__annotationsynonyms.items():
      synonym = synonym.lower()
      name = name.lower()
      if any(synonym in a.synonyms for a in result):
        raise ValueError(f"Duplicate synonym: {synonym}")
      try:
        a, = {a for a in result if name == a.name}
      except ValueError:
        raise ValueError(f"Unknown annotation for synonym: {name}")
      a.synonyms.add(synonym)
    return result

  def allowedannotation(self, nameornumber, *, logwarning=True):
    try:
      result, = {a for a in self.allowedannotations if nameornumber in {a.layer} | a.synonyms}
    except ValueError:
      typ = 'number' if isinstance(nameornumber, int) else 'name'
      raise ValueError(f"Unknown annotation {typ} {nameornumber}")
    if logwarning and nameornumber not in {result.layer, result.name}:
      self.logger.warningglobal(f"renaming annotation {nameornumber} to {result.name}")
    return result

  @methodtools.lru_cache()
  def getXMLpolygonannotations(self, *, pscale=None):
    if pscale is None:
      return self.getXMLpolygonannotations(pscale=self.pscale)

    annotations = []
    allregions = []
    allvertices = []
    annotationinfos = self.annotationinfo

    errors = []

    nodes = self.annotationnodes

    count = more_itertools.peekable(itertools.count(1))
    for node in nodes[:]:
      if not node.regions:
        if node.annotationtype != "empty":
          self.logger.warningglobal(f"Annotation {node.annotationname} is empty, skipping it")
        nodes.remove(node)
    for node in nodes:
      try:
        node.annotation = self.allowedannotation(node.annotationtype)
        node.annotationtype = node.annotation.name
      except ValueError as e:
        node.annotationerror = e

    def annotationorder(node):
      try:
        return node.annotation.layer, node.annotationsubindex
      except AttributeError:
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
        targetannotation, = {node.annotation for node in annotationnodes}
      except AttributeError:
        errors += [str(node.annotationerror) for node in annotationnodes if hasattr(node, "annotationerror")]
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
              sampleid=self.SampleID,
              layer=layeridx,
              poly="poly",
              pscale=pscale,
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
          self.logger.warning(f"Annotation {name} has the wrong color {color}, changing it to {targetcolor}")
          color = targetcolor

        annotationinfo, = (info for info in annotationinfos if info.dbname == name)
        annotationinfos.remove(annotationinfo)

        annotation = Annotation(
          color=color,
          visible=visible,
          name=name,
          sampleid=self.SampleID,
          layer=layer,
          poly="poly",
          pscale=pscale,
          apscale=self.apscale,
          annotationinfo=annotationinfo,
        )
        annotations.append(annotation)

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
                pscale=pscale,
                annotation=annotation,
              )
            )
          isNeg = region.NegativeROA

          polygon = SimplePolygon(vertices=regionvertices)
          valid = polygon.makevalid(round=True, imagescale=annotation.annoscale)

          perimeter = 0
          maxlength = 0
          longestidx = None
          for nlines, (v1, v2) in enumerate(more_itertools.pairwise(regionvertices+[regionvertices[0]]), start=1):
            length = np.sum((v1.xvec-v2.xvec)**2)**.5
            if not length: continue
            maxlength, longestidx = max((maxlength, longestidx), (length, nlines))
            perimeter += length

          saveimage = self.__saveallannotationimages
          badimage = False
          if (longestidx == 1 or longestidx == len(regionvertices)) and maxlength / (perimeter/nlines) > 30:
            self.logger.warningglobal(f"annotation polygon might not be closed: region id {regionid}")
            saveimage = True
            badimage = True

          if saveimage and self.__annotationimagefolder is not None:
            poly = SimplePolygon(vertices=regionvertices)
            if annotation.isonwsi:
              raise ValueError("saving images is not implemented on wsi")
            else:
              with QPTiff(self.qptifffilename) as fqptiff:
                zoomlevel = fqptiff.zoomlevels[0]
                img = zoomlevel[0].asarray()
            pixel = self.oneannopixel
            xymin = np.min(poly.vertexarray, axis=0).astype(units.unitdtype)
            xymax = np.max(poly.vertexarray, axis=0).astype(units.unitdtype)
            xybuffer = (xymax - xymin) / 20
            xymin -= xybuffer
            xymax += xybuffer
            (xmin, ymin), (xmax, ymax) = xymin, xymax
            fig, ax = plt.subplots(1, 1)
            plt.imshow(
              img[
                floattoint(float(ymin//pixel)):floattoint(float(ymax//pixel)),
                floattoint(float(xmin//pixel)):floattoint(float(xmax//pixel)),
              ],
              extent=[float(xmin//pixel), float(xmax//pixel), float(ymax//pixel), float(ymin//pixel)],
            )
            ax.add_patch(poly.matplotlibpolygon(fill=False, color="red", imagescale=annotation.annoscale))

            if badimage:
              openvertex1 = poly.vertexarray[0]
              openvertex2 = poly.vertexarray[{1: 1, len(regionvertices): -1}[longestidx]]
              boxxmin, boxymin = np.min([openvertex1, openvertex2], axis=0) - xybuffer/2
              boxxmax, boxymax = np.max([openvertex1, openvertex2], axis=0) + xybuffer/2
              ax.add_patch(matplotlib.patches.Rectangle((boxxmin//pixel, boxymin//pixel), (boxxmax-boxxmin)//pixel, (boxymax-boxymin)//pixel, color="violet", fill=False))

            fig.savefig(self.__annotationimagefolder/node.xmlpath.with_suffix("").with_suffix("").with_suffix(f".annotation-{regionid}.{self.__annotationimagefiletype}").name)
            plt.close(fig)

          areacutoff = node.areacutoff
          if areacutoff is not None: areacutoff = units.convertpscale(areacutoff, annotation.annoscale, pscale, power=2)
          for subpolygon in valid:
            subsubpolygons = (p for p in [subpolygon.outerpolygon] + subpolygon.subtractpolygons if not (areacutoff is not None and polygon.area < areacutoff))
            for polygon, m in zip(subsubpolygons, regioncounter): #regioncounter has to be last! https://www.robjwells.com/2019/06/help-zip-is-eating-my-iterators-items/
              if node.areacutoff is not None and polygon.area < areacutoff: continue
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
                  annoscale=annotation.annoscale,
                  pscale=pscale,
                )
              )

          m = next(regioncounter)

    if "good tissue" not in nodesbytype:
      errors.append(f"Didn't find a 'good tissue' annotation (only found: {', '.join(nodesbytype)})")
    if annotationinfos:
      errors.append(f"Extra annotationinfos: {', '.join(info.name for info in annotationinfos)}")

    if errors:
      raise ValueError("\n".join(errors))

    return annotations, allregions, allvertices

class XMLPolygonAnnotationReaderStandalone(XMLPolygonAnnotationReader):
  def __init__(self, infofile, *, pscale=None, apscale=None, logger=dummylogger, **kwargs):
    self.__infofile = infofile
    self.__logger = logger
    super().__init__(**kwargs)
    if pscale is None: pscale = 1
    if apscale is None:
      if self.annotationimagefolder is not None:
        with QPTiff(self.qptifffilename) as fqptiff:
          apscale = fqptiff.apscale
      else:
        apscale = 1
    self.__pscale = pscale
    self.__apscale = apscale
  @property
  def pscale(self): return self.__pscale
  @property
  def apscale(self): return self.__apscale

  @property
  def logger(self): return self.__logger

  @property
  def annotationinfofile(self): return self.__infofile
  @property
  def scanfolder(self): return self.annotationinfofile.parent

  @property
  def SampleID(self): return 0

class XMLPolygonAnnotationReaderWithOutline(XMLPolygonAnnotationReader, TissueMaskLoaderWithPolygons):
  def getannotationnode(self, info):
    if info.isfrommask:
      return AnnotationNodeFromPolygons("outline", self.tissuemaskpolygons(), color=self.allowedannotation("outline").color, annoscale=self.pscale, areacutoff=self.tissuemaskpolygonareacutoff())
    return super().getannotationnode(info)

class XMLPolygonAnnotationFileInfoWriter(XMLPolygonAnnotationFileBase, ThingWithLogger):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @property
  @abc.abstractmethod
  def annotationsource(self): pass
  @property
  @abc.abstractmethod
  def annotationposition(self): pass
  @property
  @abc.abstractmethod
  def scanfolder(self): pass
  @property
  @abc.abstractmethod
  def SampleID(self): pass

  @methodtools.lru_cache()
  def getannotationinfo(self, *, log=False):
    logger = self.logger if log else dummylogger
    xmlfile = self.annotationspolygonsxmlfile
    with open(xmlfile, "rb") as f:
      nodes = jxmlease.parse(f)["Annotations"]["Annotation"]
      if isinstance(nodes, jxmlease.XMLDictNode): nodes = [nodes]
      nodedict = {node.get_xml_attr("Name").lower(): node for node in nodes}
      if len(nodes) != len(nodedict):
        raise ValueError(f"Duplicate annotation names in {xmlfile}: {collections.Counter(node.get_xml_attr('Name').lower() for node in nodes)}")

    with open(xmlfile, "rb") as f:
      hash = hashlib.sha256()
      hash.update(f.read())
      xmlsha = hash.hexdigest()

    message = f"xml file {xmlfile.name}, hash {xmlsha}, contains annotations drawn on the {self.annotationsource}"
    if self.annotationposition is not None:
      message += f", image position {(self.annotationposition / self.onepixel).astype(float)} pixels"
    logger.info(message)

    annotationinfos = [
      AnnotationInfo(
        sampleid=self.SampleID,
        originalname=name,
        dbname=name,
        annotationsource=self.annotationsource,
        position=self.annotationposition,
        pscale=self.pscale,
        apscale=self.apscale,
        xmlfile=xmlfile.name,
        xmlsha=xmlsha,
        scanfolder=self.scanfolder,
      ) for name in nodedict
    ]

    return annotationinfos

  @property
  def annotationinfo(self):
    xmlfile = self.annotationspolygonsxmlfile
    newinfo = self.getannotationinfo()

    newfile, = {i.xmlfile for i in newinfo}
    newsha, = {i.xmlsha for i in newinfo}
    assert newfile == xmlfile.name

    try:
      oldinfo = super().annotationinfo
    except FileNotFoundError:
      return newinfo
    else:
      try:
        oldfile, = {i.xmlfile for i in oldinfo}
        oldsha, = {i.xmlsha for i in oldinfo}
      except ValueError:
        raise ValueError(f"AnnotationInfos in {oldfile} are not all from the same file or version of the file")
      if oldfile != newfile:
        raise ValueError(f"AnnotationInfos in {xmlfile} are from the wrong filename {oldfile}")
      if oldsha != newsha:
        raise ValueError(f"AnnotationInfos in {xmlfile} are from a different version of the file with hash {oldsha}, current hash is {xmlfile.xmlsha}.")

      try:
        for old, new in more_itertools.zip_equal(oldinfo, newinfo):
          if old != new:
            raise ValueError(f"AnnotationInfos are not consistent with the xml file:\n{new}\n{old}")
      except more_itertools.UnequalIterablesError:
        raise ValueError(f"AnnotationInfos are not consistent with the xml file ({len(newinfo)} annotations, {len(oldinfo)} infos)")

      return oldinfo

  def writeannotationinfos(self):
    self.getannotationinfo(log=True)
    return writetable(self.annotationinfofile, self.annotationinfo)

class XMLPolygonAnnotationFileInfoWriterStandalone(XMLPolygonAnnotationFileInfoWriter):
  def __init__(self, *, infofile=None, xmlfile, annotationsource, annotationposition=None, pscale, apscale, logger=dummylogger, **kwargs):
    self.__infofile = infofile
    self.__xmlfile = xmlfile
    self.__annotationsource = annotationsource
    self.__annotationposition = annotationposition
    self.__pscale = pscale
    self.__apscale = apscale
    self.__logger = logger
    super().__init__(**kwargs)
  @property
  def scanfolder(self):
    return self.annotationspolygonsxmlfile.parent
  @property
  def annotationspolygonsxmlfile(self):
    return self.__xmlfile
  @property
  def annotationinfofile(self):
    result = self.__infofile
    if result is None: return super().annotationinfofile
    return result

  @property
  def logger(self): return self.__logger

  @property
  def annotationsource(self): return self.__annotationsource
  @property
  def annotationposition(self): return self.__annotationposition
  @property
  def pscale(self): return self.__pscale
  @property
  def apscale(self): return self.__apscale
  @property
  def SampleID(self): return 0

def writeannotationinfostandalone(*, infofile, xmlfile, **kwargs):
  writer = XMLPolygonAnnotationFileInfoWriterStandalone(infofile=infofile, xmlfile=xmlfile, **kwargs)
  writer.writeannotationinfos()
  return writer.annotationinfo

def writeannotationcsvsstandalone(dbloadfolder, infofile, csvprefix=None, **kwargs):
  dbloadfolder = pathlib.Path(dbloadfolder)
  dbloadfolder.mkdir(parents=True, exist_ok=True)
  annotations, regions, vertices = XMLPolygonAnnotationReaderStandalone(infofile=infofile, **kwargs).getXMLpolygonannotations()
  if csvprefix is None:
    csvprefix = ""
  elif csvprefix.endswith("_"):
    pass
  else:
    csvprefix = csvprefix+"_"
  writetable(dbloadfolder/f"{csvprefix}annotations.csv", annotations)
  writetable(dbloadfolder/f"{csvprefix}regions.csv", regions)
  writetable(dbloadfolder/f"{csvprefix}vertices.csv", vertices)

def add_rename_annotation_argument(argumentparser):
  argumentparser.add_argument("--rename-annotation", nargs=2, action=ArgParseAddToDict, dest="annotationsynonyms", metavar=("XMLNAME", "NEWNAME"), help="Rename an annotation given in the xml file to a new name (which has to be in the master list)")

def writeannotationinfo(args=None):
  p = argparse.ArgumentParser(description="read an annotations.polygons.xml file and write out the annotation info csv file")
  p.add_argument("xmlfile", type=pathlib.Path, help="path to the annotations.polygons.xml file")
  p.add_argument("--infofile", type=pathlib.Path, help="output path for the annotation info (default: xmlfile with the suffix .csv)")
  g = p.add_mutually_exclusive_group(required=True)
  g.add_argument("--annotations-on-wsi", action="store_const", dest="annotationsource", const="wsi", help="annotations were drawn on the AstroPath image")
  g.add_argument("--annotations-on-qptiff", action="store_const", dest="annotationsource", const="qptiff", help="annotations were drawn on the qptiff")
  p.add_argument("--annotation-position", type=float, dest="annotationposition", nargs=2, help="position of the wsi when the annotations were drawn")
  args = p.parse_args(args=args)
  with units.setup_context("fast"):
    return writeannotationinfostandalone(**args.__dict__, logger=printlogger("annotations"), pscale=1, apscale=1)

def writeannotationcsvs(args=None):
  p = argparse.ArgumentParser(description="read an annotations.polygons.xml file and write out csv files for the annotations, regions, and vertices")
  p.add_argument("dbloadfolder", type=pathlib.Path, help="folder to write the output csv files in")
  p.add_argument("infofile", type=pathlib.Path, help="path to the annotation info csv")
  p.add_argument("--csvprefix", help="prefix to put in front of the csv file names")
  add_rename_annotation_argument(p)
  args = p.parse_args(args=args)
  with units.setup_context("fast"):
    return writeannotationcsvsstandalone(**args.__dict__, logger=printlogger("annotations"))

def checkannotations(args=None):
  p = argparse.ArgumentParser(description="run astropath checks on an annotations.polygons.xml file")
  p.add_argument("infofile", type=pathlib.Path, help="path to the annotation info csv")
  g = p.add_mutually_exclusive_group()
  g.add_argument("--save-polygon-images", action="store_const", dest="annotationimagefolder", const=pathlib.Path("."), help="save all annotation images to the current folder")
  g.add_argument("--save-polygon-images-folder", type=pathlib.Path, dest="annotationimagefolder", help="save all annotation images to the given directory")
  g.add_argument("--save-bad-polygon-images", action="store_const", dest="badannotationimagefolder", const=pathlib.Path("."), help="if there are unclosed annotations, save a debug image to the current directory pointing out the problem")
  g.add_argument("--save-bad-polygon-images-folder", type=pathlib.Path, dest="badannotationimagefolder", help="if there are unclosed annotations, save a debug image to the given directory pointing out the problem")
  p.add_argument("--save-images-filetype", default="pdf", choices=("pdf", "png"), dest="annotationimagefiletype", help="image format to save debug images")
  add_rename_annotation_argument(p)
  args = p.parse_args(args=args)
  if args.annotationimagefolder is not None:
    args.saveallannotationimages = True
  else:
    args.annotationimagefolder = args.badannotationimagefolder
  del args.badannotationimagefolder
  logger = printlogger("annotations")
  with units.setup_context("fast"):
    XMLPolygonAnnotationReaderStandalone(**args.__dict__, logger=logger).getXMLpolygonannotations()
  logger.info(f"{args.infofile} looks good!")
