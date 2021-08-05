import argparse, itertools, jxmlease, matplotlib.patches, matplotlib.pyplot as plt, methodtools, more_itertools, numpy as np, pathlib
from ..utilities import units
from ..utilities.dataclasses import MetaDataAnnotation, MyDataClassFrozen
from ..utilities.misc import floattoint
from ..utilities.tableio import readtable, writetable
from .csvclasses import Annotation, Region, Vertex
from .logging import dummylogger, printlogger
from .polygon import SimplePolygon
from .qptiff import QPTiff

class AllowedAnnotation(MyDataClassFrozen):
  name: str = MetaDataAnnotation(readfunction=str.lower)
  layer: int
  color: str
  synonyms: set = MetaDataAnnotation(set(), readfunction=lambda x: set(x.lower().split(",")) if x else set(), writefunction=lambda x: ",".join(sorted(x)))

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

  @staticmethod
  def annotationname(xmlnode):
    return xmlnode.get_xml_attr("Name").lower().strip()

  def getXMLpolygonannotations(self):
    annotations = []
    allregions = []
    allvertices = []

    errors = []

    with open(self.xmlfile, "rb") as f:
      count = more_itertools.peekable(itertools.count(1))
      nodes = [node for _, _, node in jxmlease.parse(f, generator="/Annotations/Annotation")]
      def annotationorder(node):
        try:
          return self.allowedannotation(self.annotationname(node), logwarning=False).layer
        except ValueError:
          return float("inf")
      sortednodes = sorted(nodes, key=annotationorder)
      if sortednodes != nodes:
        message = f"Annotations are in the wrong order: target order is {', '.join(_.name for _ in self.allowedannotations)}, but your order is {', '.join(self.annotationname(node) for node in nodes)}."
        if self.__reorderannotations:
          self.__logger.warning(message+"  Reordering them.")
        else:
          errors.append(message)
      nodes = sortednodes

      for layer, node in zip(count, nodes):
        color = f"{int(node.get_xml_attr('LineColor')):06X}"
        color = color[4:6] + color[2:4] + color[0:2]
        visible = {
          "true": True,
          "false": False,
        }[node.get_xml_attr("Visible").lower().strip()]
        name = self.annotationname(node)
        if name == "empty":
          count.prepend(layer)
          continue
        try:
          targetannotation = self.allowedannotation(name)
        except ValueError as e:
          errors.append(str(e))
          continue
        name = targetannotation.name
        targetlayer = targetannotation.layer
        targetcolor = targetannotation.color
        if layer > targetlayer:
          assert False
        else:
          while layer < targetlayer:
            emptycolor = self.allowedannotation(layer).color
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

        if not node["Regions"]: continue
        regions = node["Regions"]["Region"]
        if isinstance(regions, jxmlease.XMLDictNode): regions = regions,
        m = 1
        for region in regions:
          regioncounter = itertools.count(m)
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
          isNeg = bool(int(region.get_xml_attr("NegativeROA")))

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

          bigpolygon = polygon
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
                  type=region.get_xml_attr("Type"),
                  nvert=len(regionvertices),
                  poly=None,
                  apscale=self.apscale,
                  pscale=self.pscale,
                )
              )

          m = next(regioncounter)

    if not any(a.name == "good tissue" for a in annotations):
      errors.append(f"Didn't find a 'good tissue' annotation (only found: {', '.join(_.name for _ in annotations if _.name != 'empty')})")

    if errors:
      raise ValueError("\n".join(errors))

    return annotations, allregions, allvertices

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
  with units.setup_context("fast"):
    XMLPolygonAnnotationReader(**args.__dict__, logger=printlogger("annotations")).getXMLpolygonannotations()
  print(f"{args.xmlfile} looks good!")
