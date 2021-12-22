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
    g.add_argument("--annotations-on-wsi", action="store_true", dest="annotationsonwsi", help="annotations were drawn on the AstroPath image")
    g.add_argument("--annotations-on-qptiff", action="store_false", dest="annotationsonwsi", help="annotations were drawn on the qptiff")
    p.add_argument("--annotation-position", nargs=2, type=float)
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    if parsed_args_dict["annotation_position"] is not None and not parsed_args_dict["annotationsonwsi"]:
      raise ValueError("--annotation-position is only valid for --annotations-on-wsi")
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "annotationsonwsi": parsed_args_dict.pop("annotationsonwsi"),
      "annotationposition": parsed_args_dict.pop("annotation_position"),
    }

class AnnotationInfoReaderSample(DbloadSample, units.ThingWithAnnoscale):
  def __init__(self, *args, annotationsonwsi=None, annotationposition=None, **kwargs):
    self.__annotationsonwsi = annotationsonwsi
    if annotationposition is not None: annotationposition = np.array(annotationposition)
    self.__annotationposition = annotationposition
    super().__init__(*args, **kwargs)

  def readannotationinfo(self, *, check_compatibility=True):
    try:
      annotationinfo = self.readcsv("annotationinfo", Constant)
    except FileNotFoundError:
      dct = {}
    else:
      dct = {_.name: _.value for _ in annotationinfo}

    annotationsonwsi = dct.pop("annotationsonwsi", None)
    if annotationsonwsi is not None: annotationsonwsi = bool(annotationsonwsi)
    annotationxposition = dct.pop("annotationxposition", None)
    annotationyposition = dct.pop("annotationyposition", None)
    if dct:
      raise ValueError(f"Unknown fields {', '.join(dct)} in annotationinfo.csv")
    if annotationxposition is not None is not annotationyposition:
      annotationposition = np.array([annotationxposition, annotationyposition])
    elif annotationxposition is annotationyposition is None:
      annotationposition = None
    else:
      raise ValueError("One of annotationxposition and annotationyposition is present in annotationinfo.csv, but not both")      
    if check_compatibility:
      if annotationsonwsi is not None is not self.__annotationsonwsi != annotationsonwsi:
        raise ValueError(f"annotationsonwsi from constructor {self.__annotationsonwsi} doesn't match the one from readannotationinfo {annotationsonwsi}")
      if annotationposition is not None is not self.__annotationposition != annotationposition:
        raise ValueError(f"annotationposition from constructor {self.__annotationposition} doesn't match the one from readannotationinfo {annotationposition}")

    if annotationsonwsi is None:
      annotationsonwsi = self.__annotationsonwsi
    if annotationsonwsi is None:
      raise ValueError("No annotationsonwsi given or found in annotationinfo.csv")

    if annotationsonwsi:
      if annotationposition is None:
        if annotationposition is None:
          annotationposition = self.__annotationposition
        raise ValueError("No annotationposition given or found in annotationinfo.csv")

    if not annotationsonwsi and annotationposition is not None:
      raise ValueError("annotationposition is meant for when the annotations are drawn on the wsi")

    self.__annotationsonwsi = annotationsonwsi
    self.__annotationposition = annotationposition

  @property
  def annotationsonwsi(self): return self.__annotationsonwsi
  @property
  def annotationposition(self): return self.__annotationposition
  @annotationposition.setter
  def annotationposition(self, annotationposition): self.__annotationposition = annotationposition
  @property
  def annoscale(self):
    """
    Scale of the annotations
    """
    return self.constantsdict.get("annoscale", self.pscale / 2 if self.annotationsonwsi else self.apscale)


class AnnotationInfoWriterSample(AnnotationInfoReaderSample, AnnotationInfoWriterArgumentParser):
  def writeannotationinfo(self):
    pscales = {"pscale": self.pscale, "apscale": self.apscale, "qpscale": self.qpscale}
    annotationinfo = [
      Constant(
        name="annotationsonwsi",
        value=self.annotationsonwsi,
        **pscales,
      ),
    ]
    if self.annotationsonwsi:
      annotationinfo += [
        Constant(
          name="annotationxposition",
          value=self.annotationposition[0],
          **pscales,
        ),
        Constant(
          name="annotationyposition",
          value=self.annotationposition[1],
          **pscales,
        ),
      ]
    self.writecsv("annotationinfo", annotationinfo)

class AnnotationInfoWriterCohort(DbloadCohort, AnnotationInfoWriterArgumentParser):
  def __init__(self, *args, annotationsonwsi, annotationposition, **kwargs):
    self.__annotationsonwsi = annotationsonwsi
    self.__annotationposition = annotationposition
    super().__init__(*args, **kwargs)

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "annotationsonwsi": self.__annotationsonwsi,
      "annotationposition": self.__annotationposition,
    }

class MergeAnnotationXMLsArgumentParser(AnnotationInfoWriterArgumentParser):
  require_annotations_on_wsi_or_qptiff_argument = True

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--annotation", nargs=2, action=ArgParseAddRegexToDict, metavar=("ANNOTATION", "FILENAME_REGEX"), default={}, help="take annotation with this name from the annotation file that matches the regex", dest="annotationselectiondict")
    p.add_argument("--skip-annotation", action="append", metavar="ANNOTATION", default=[], help="skip this annotation if it exists in an xml file")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "annotationselectiondict": parsed_args_dict.pop("annotationselectiondict"),
      "skipannotations": parsed_args_dict.pop("skip_annotation"),
    }

class MergeAnnotationXMLsSample(AnnotationInfoWriterSample, WorkflowSample, MergeAnnotationXMLsArgumentParser):
  def __init__(self, *args, annotationselectiondict, skipannotations, **kwargs):
    self.__annotationselectiondict = annotationselectiondict
    self.__skipannotations = skipannotations
    super().__init__(*args, **kwargs)
    duplicates = frozenset(self.__annotationselectiondict) & frozenset(self.__skipannotations)
    if duplicates:
      raise ValueError(f"Some annotations are specified to be taken from an xml file and to be skipped: {', '.join(duplicates)}")

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
