import collections, contextlib, jxmlease, methodtools, numpy as np, re
from ...shared.annotationpolygonxmlreader import XMLPolygonAnnotationFileInfoWriter
from ...shared.argumentparser import DbloadArgumentParser
from ...shared.cohort import DbloadCohort, XMLPolygonFileCohort, WorkflowCohort
from ...shared.csvclasses import AnnotationInfo
from ...shared.sample import WorkflowSample, XMLPolygonAnnotationSample
from ...utilities import units
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.misc import ArgParseAddRegexToDict, ArgParseAddToDict, ArgParseAddTupleToDict
from ...utilities.tableio import writetable
from ..align.alignsample import AlignSample, ReadAffineShiftSample

class AnnotationInfoWriterArgumentParser(DbloadArgumentParser):
  def __init__(self, *args, annotationsource, annotationposition, annotationpositionfromaffineshift, **kwargs):
    providedposition = annotationposition is not None or annotationpositionfromaffineshift
    if annotationsource == "wsi":
      if not providedposition:
        raise ValueError("For wsi annotations, have to provide the annotation position or specify that it's taken from the affine shift")
    elif annotationsource == "qptiff":
      if providedposition:
        raise ValueError("Don't provide annotationposition for qptiff")
    else:
      assert False

    self.__annotationsource = annotationsource
    if annotationposition is not None:
      annotationposition = np.array(annotationposition, dtype=units.unitdtype)
    self.__annotationposition = annotationposition
    self.__annotationpositionfromaffineshift = annotationpositionfromaffineshift
    super().__init__(*args, **kwargs)

  @property
  def annotationsource(self): return self.__annotationsource
  @property
  def annotationposition(self): return self.__annotationposition
  @property
  def annotationpositionfromaffineshift(self): return self.__annotationpositionfromaffineshift

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--annotation-position", nargs=2, metavar=("XPOS", "YPOS"), help="position of the annotations if they were drawn on the wsi", dest="annotationposition", type=float)
    g.add_argument("--annotation-position-from-affine-shift", action="store_true", dest="annotationpositionfromaffineshift")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--annotations-on-wsi", action="store_const", dest="annotationsource", const="wsi", help="Unless otherwise specified, annotations are drawn on the wsi")
    g.add_argument("--annotations-on-qptiff", action="store_const", dest="annotationsource", const="qptiff", help="Unless otherwise specified, annotations are drawn on the qptiff")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    annotationsource = parsed_args_dict.pop("annotationsource")
    annotationposition = parsed_args_dict.pop("annotationposition")
    annotationpositionfromaffineshift = parsed_args_dict.pop("annotationpositionfromaffineshift")

    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "annotationsource": annotationsource,
      "annotationposition": annotationposition,
      "annotationpositionfromaffineshift": annotationpositionfromaffineshift,
    }

class WriteAnnotationInfoSample(ReadAffineShiftSample, XMLPolygonAnnotationSample, WorkflowSample, AnnotationInfoWriterArgumentParser, XMLPolygonAnnotationFileInfoWriter):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def run(self):
    self.writeannotationinfos()

  @methodtools.lru_cache()
  @property
  def annotationposition(self):
    result = super().annotationposition
    if result is None:
      if self.annotationsource == "wsi":
        assert self.annotationpositionfromaffineshift
        return self.affineshift
      elif self.annotationsource == "qptiff":
        return None
      else:
        assert False
    else:
      if self.annotationsource == "wsi":
        pass
      elif self.annotationsource == "qptiff":
        assert False
      else:
        assert False
      return result * self.onepixel

  @classmethod
  def getoutputfiles(cls, **kwargs):
    annotationsxmlregex = kwargs["annotationsxmlregex"]
    im3root = kwargs["im3root"]
    Scan = kwargs["Scan"]
    SlideID = kwargs["SlideID"]
    xmls = [
      _ for _ in (im3root/SlideID/"im3"/f"Scan{Scan}").glob(f"*{SlideID}*annotations..polygons*.xml")
      if annotationsxmlregex is None or re.match(annotationsxmlregex, _.name)
    ]
    return [
      *super().getoutputfiles(**kwargs),
      *(xml.with_suffix(".annotationinfo.csv") for xml in xmls)
    ]

  def inputfiles(self, **kwargs):
    result = [
      *super().inputfiles(**kwargs),
    ]
    if self.annotationpositionfromaffineshift:
      result += [
        self.csv("affine"),
      ]
    return result

  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    result = [
      *super().workflowdependencyclasses(**kwargs),
    ]
    if kwargs["annotationpositionfromaffineshift"]:
      result += [
        AlignSample,
      ]
    return result

  @classmethod
  def logmodule(cls): return "writeannotationinfo"

  @property
  def workflowkwargs(self):
    return {
      **super().workflowkwargs,
      "annotationpositionfromaffineshift": self.annotationpositionfromaffineshift,
    }

class WriteAnnotationInfoCohort(DbloadCohort, XMLPolygonFileCohort, WorkflowCohort, AnnotationInfoWriterArgumentParser):
  sampleclass = WriteAnnotationInfoSample

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "annotationsource": self.annotationsource,
      "annotationposition": self.annotationposition,
      "annotationpositionfromaffineshift": self.annotationpositionfromaffineshift,
    }

  @property
  def workflowkwargs(self):
    return {
      **super().workflowkwargs,
      "annotationpositionfromaffineshift": self.annotationpositionfromaffineshift,
    }

class MergeAnnotationXMLsArgumentParser(AnnotationInfoWriterArgumentParser, DbloadArgumentParser):
  def __init__(self, *args, annotationselectiondict, skipannotations, **kwargs):
    self.__annotationselectiondict = annotationselectiondict
    self.__skipannotations = skipannotations
    super().__init__(*args, **kwargs)

    duplicates = frozenset(self.annotationselectiondict) & frozenset(self.skipannotations)
    if duplicates:
      raise ValueError(f"Some annotations are specified to be taken from an xml file and to be skipped: {', '.join(duplicates)}")

    nosource = set()
    for _ in self.annotationselectiondict:
      try:
        self.annotationsourcedict[_]
      except KeyError:
        nosource.add(_)
    if nosource:
      raise ValueError(f"Some annotations are not specified to come from the wsi or qptiff: {', '.join(nosource)}")

  @property
  def annotationselectiondict(self): return self.__annotationselectiondict
  @property
  def skipannotations(self): return self.__skipannotations

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--annotation", nargs=2, action=ArgParseAddRegexToDict, metavar=("ANNOTATION", "FILENAME_REGEX"), default={}, help="take annotation with this name from the annotation file that matches the regex", dest="annotationselectiondict", case_sensitive=False)
    p.add_argument("--skip-annotation", action="append", metavar="ANNOTATION", default=[], help="skip this annotation if it exists in an xml file")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "annotationselectiondict": parsed_args_dict.pop("annotationselectiondict"),
      "skipannotations": parsed_args_dict.pop("skip_annotation"),
    }

class MergeAnnotationXMLsSample(WorkflowSample, MergeAnnotationXMLsArgumentParser):
  @property
  def xmloutput(self):
    return self.scanfolder/f"{self.SlideID}_Scan{self.Scan}.annotations.polygons.merged.xml"

  @classmethod
  def getoutputfiles(cls, SlideID, *, im3root, Scan, **kwargs):
    return [
      *super().getoutputfiles(SlideID=SlideID, **kwargs),
      im3root/SlideID/UNIV_CONST.IM3_DIR_NAME/f"Scan{Scan}"/f"{SlideID}_Scan{Scan}.annotations.polygons.merged.xml",
    ]

  @classmethod
  def logmodule(cls):
    return "mergeannotationxmls"

  @methodtools.lru_cache()
  @property
  def allxmls(self):
    return frozenset(
      filename
      for filename in self.scanfolder.glob(f"*{self.SlideID}*annotations.polygons*.xml")
      if filename != self.xmloutput
    )

  @methodtools.lru_cache()
  @property
  def annotationxmldict(self):
    result = {}
    for annotation, regex in self.annotationselectiondict.items():
      regex = re.compile(regex)
      filenames = {xml for xml in self.allxmls if regex.match(xml.name)}
      try:
        filename, = filenames
      except ValueError:
        if filenames:
          raise IOError(f"Found multiple xmls matching {regex.pattern}: " + ", ".join(_.name for _ in filenames))
        else:
          raise FileNotFoundError(f"Didn't find any xmls matching {regex.pattern}")
      result[annotation] = filename
    return result

  def mergexmls(self):
    with contextlib.ExitStack() as stack:
      xml = {}
      allnames = set()
      for xmlfile in self.allxmls:
        f = stack.enter_context(open(xmlfile, "rb"))
        nodes = jxmlease.parse(f)["Annotations"]["Annotation"]
        if isinstance(nodes, jxmlease.XMLDictNode): nodes = [nodes]
        nodedict = {node.get_xml_attr("Name").lower(): node for node in nodes}
        if len(nodes) != len(nodedict):
          raise ValueError(f"Duplicate annotation names in {xmlfile.name}: {collections.Counter(node.get_xml_attr('Name').lower() for node in nodes)}")
        xml[xmlfile] = nodedict
        allnames.update(nodedict.keys())

      unknown = allnames - set(self.annotationxmldict) - set(self.skipannotations)
      if unknown:
        raise ValueError(f"Found unknown annotations in xml files: {', '.join(sorted(unknown))}")

      annotations = []
      for name, xmlfile in self.annotationxmldict.items():
        self.logger.info(f"Taking {name} from {xmlfile.name}")
        annotations.append(xml[xmlfile][name.lower()])

      for name in sorted(set(self.skipannotations) & allnames):
        self.logger.info(f"Skipping {name}")

      result = jxmlease.XMLDictNode({"Annotations": jxmlease.XMLDictNode({"Annotation": jxmlease.XMLListNode(annotations)})})
      with open(self.xmloutput, "w") as f:
        f.write(result.emit_xml())

    names = [node.get_xml_attr("Name").lower() for node in annotations]
    self.writeannotationinfo(names)

  def run(self, **kwargs):
    return self.mergexmls(**kwargs)

class MergeAnnotationXMLsCohort(WorkflowCohort, MergeAnnotationXMLsArgumentParser):
  sampleclass = MergeAnnotationXMLsSample
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "annotationselectiondict": self.annotationselectiondict.copy(),
      "skipannotations": self.skipannotations.copy(),
    }

def mergeannotationxmlssample(*args, **kwargs):
  MergeAnnotationXMLsSample.runfromargumentparser(*args, **kwargs)

def mergeannotationxmlscohort(*args, **kwargs):
  MergeAnnotationXMLsCohort.runfromargumentparser(*args, **kwargs)

def writeannotationinfosample(*args, **kwargs):
  WriteAnnotationInfoSample.runfromargumentparser(*args, **kwargs)

def writeannotationinfocohort(*args, **kwargs):
  WriteAnnotationInfoCohort.runfromargumentparser(*args, **kwargs)
