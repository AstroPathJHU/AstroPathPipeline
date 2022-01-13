import collections, contextlib, jxmlease, methodtools, numpy as np, re
from ...shared.argumentparser import DbloadArgumentParser
from ...shared.cohort import DbloadCohort, WorkflowCohort
from ...shared.csvclasses import Constant
from ...shared.sample import DbloadSample, WorkflowSample
from ...utilities import units
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.misc import ArgParseAddRegexToDict

class AnnotationInfoWriterArgumentParser(DbloadArgumentParser):
  require_annotations_on_wsi_or_qptiff_argument = False

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    g = p.add_mutually_exclusive_group(required=cls.require_annotations_on_wsi_or_qptiff_argument)
    g.add_argument("--annotations-on-wsi", action="store_const", dest="annotationsource", help="annotations were drawn on the AstroPath image", const="wsi")
    g.add_argument("--annotations-on-qptiff", action="store_const", dest="annotationsource", help="annotations were drawn on the qptiff", const="qptiff")
    g.add_argument("--annotations-on-both", action="store_const", dest="annotationsource", help="annotations were drawn on either the wsi or qptiff or a mix, and the annotations.csv already exists with that information", const="both")
    p.add_argument("--annotation-position", nargs=2, type=float)
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    if parsed_args_dict["annotation_position"] is not None and not parsed_args_dict["annotationsonwsi"]:
      raise ValueError("--annotation-position is only valid for --annotations-on-wsi")
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "annotationsource": parsed_args_dict.pop("annotationsource"),
      "annotationposition": parsed_args_dict.pop("annotation_position"),
    }

class MergeAnnotationXMLsArgumentParser(DbloadArgumentParser):
  def __init__(self, *args, annotationselectiondict, skipannotations, **kwargs):
    self.__annotationselectiondict = annotationselectiondict
    self.__annotationsourcedict = annotationsourcedict
    self.__annotationpositiondict = annotationpositiondict
    self.__skipannotations = skipannotations
    super().__init__(*args, **kwargs)

    duplicates = frozenset(self.__annotationselectiondict) & frozenset(self.__skipannotations)
    if duplicates:
      raise ValueError(f"Some annotations are specified to be taken from an xml file and to be skipped: {', '.join(duplicates)}")
    nosource = frozenset(self.__annotationselectiondict) - frozenset(self.__annotationsourcedict)
    if nosource:
      raise ValueError(f"Some annotations are not specified to come from the wsi or qptiff: {', '.join(nosource)}")

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--annotation", nargs=2, action=ArgParseAddRegexToDict, metavar=("ANNOTATION", "FILENAME_REGEX"), default={}, help="take annotation with this name from the annotation file that matches the regex", dest="annotationselectiondict")
    p.add_argument("--annotation-source", nargs=2, action=ArgParseAddToDict, metavar=("ANNOTATION", "SOURCE"), default={}, help="source of the annotations (wsi or qptiff)", dest="annotationsourcedict")
    p.add_argument("--annotation-position", nargs=3, action=ArgParseAddToDict, metavar=("ANNOTATION", "XPOS", "YPOS"), default={}, help="position of the annotations if they were drawn on the wsi", dest="annotationpositiondict")
    p.add_argument("--skip-annotation", action="append", metavar="ANNOTATION", default=[], help="skip this annotation if it exists in an xml file")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "annotationselectiondict": parsed_args_dict.pop("annotationselectiondict"),
      "annotationsourcedict": parsed_args_dict.pop("annotationsourcedict"),
      "annotationpositiondict": parsed_args_dict.pop("annotationpositiondict"),
      "skipannotations": parsed_args_dict.pop("skip_annotation"),
    }

class MergeAnnotationXMLsSample(AnnotationInfoWriterSample, WorkflowSample, MergeAnnotationXMLsArgumentParser):
  @property
  def xmloutput(self):
    return self.scanfolder/f"{self.SlideID}_Scan{self.Scan}.annotations.polygons.merged.xml"

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, im3root, Scan, **otherworkflowkwargs):
    return [
      im3root/SlideID/UNIV_CONST.IM3_DIR_NAME/f"Scan{Scan}"/f"{SlideID}_Scan{Scan}.annotations.polygons.merged.xml",
      dbloadroot/SlideID/"dbload"/f"{SlideID}_annotationinfo.csv",
    ]
  def inputfiles(self, **kwargs):
    return []

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
  def annotationselectiondict(self):
    result = {}
    for annotation, regex in self.__annotationselectiondict.items():
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
          raise ValueError(f"Duplicate annotation names in {xmlfile.name}: {collections.Counter(node.get_xml_attr('Name') for node in nodes)}")
        xml[xmlfile] = nodedict
        allnames.update(nodedict.keys())

      unknown = allnames - set(self.annotationselectiondict) - set(self.__skipannotations)
      if unknown:
        raise ValueError(f"Found unknown annotations in xml files: {', '.join(sorted(unknown))}")

      annotations = []
      for name, xmlfile in self.annotationselectiondict.items():
        self.logger.info(f"Taking {name} from {xmlfile.name}")
        annotations.append(xml[xmlfile][name.lower()])

      for name in sorted(set(self.__skipannotations) & allnames):
        self.logger.info(f"Skipping {name}")

      result = jxmlease.XMLDictNode({"Annotations": jxmlease.XMLDictNode({"Annotation": jxmlease.XMLListNode(annotations)})})
      with open(self.xmloutput, "w") as f:
        f.write(result.emit_xml())

    self.writeannotationinfo()

  def run(self, **kwargs):
    return self.mergexmls(**kwargs)

  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return []

class MergeAnnotationXMLsCohort(AnnotationInfoWriterCohort, WorkflowCohort, MergeAnnotationXMLsArgumentParser):
  sampleclass = MergeAnnotationXMLsSample
  def __init__(self, *args, annotationselectiondict, skipannotations, **kwargs):
    self.__annotationselectiondict = annotationselectiondict
    self.__skipannotations = skipannotations
    super().__init__(*args, **kwargs)

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "annotationselectiondict": self.__annotationselectiondict,
      "skipannotations": self.__skipannotations,
    }

def samplemain(*args, **kwargs):
  MergeAnnotationXMLsSample.runfromargumentparser(*args, **kwargs)

def cohortmain(*args, **kwargs):
  MergeAnnotationXMLsCohort.runfromargumentparser(*args, **kwargs)
