import collections, contextlib, jxmlease, methodtools, re
from ...shared.argumentparser import DbloadArgumentParser
from ...shared.cohort import DbloadCohort, WorkflowCohort
from ...shared.csvclasses import AnnotationInfo
from ...shared.sample import DbloadSample, WorkflowSample
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.misc import ArgParseAddRegexToDict, ArgParseAddToDict, ArgParseAddTupleToDict

class AnnotationInfoWriterArgumentParser(DbloadArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--annotation-source", nargs=2, action=ArgParseAddToDict, metavar=("ANNOTATION", "SOURCE"), default={}, help="source of the annotations (wsi or qptiff)", dest="annotationsourcedict")
    p.add_argument("--annotation-position", nargs=3, action=ArgParseAddTupleToDict, metavar=("ANNOTATION", "XPOS", "YPOS"), default={}, help="position of the annotations if they were drawn on the wsi", dest="annotationpositiondict")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--annotations-on-wsi", action="store_const", dest="defaultsource", const="wsi", help="Unless otherwise specified, annotations are drawn on the wsi")
    g.add_argument("--annotations-on-qptiff", action="store_const", dest="defaultsource", const="qptiff", help="Unless otherwise specified, annotations are drawn on the qptiff")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    defaultsource = parsed_args_dict.pop("defaultsource")
    annotationsourcedict = parsed_args_dict.pop("annotationsourcedict")
    if defaultsource is not None:
      annotationsourcedict = collections.defaultdict(lambda: defaultsource, annotationsourcedict)

    annotationpositiondict = parsed_args_dict.pop("annotationpositiondict")
    annotationpositiondict = collections.defaultdict(lambda: None, annotationpositiondict)

    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "annotationsourcedict": annotationsourcedict,
      "annotationpositiondict": annotationpositiondict,
    }

class AnnotationInfoWriterSampleBase(DbloadSample, AnnotationInfoWriterArgumentParser):
  def writeannotationinfo(self, names):
    annotationinfos = [
      AnnotationInfo(
        sampleid=self.SampleID,
        name=name,
        isonwsi={"wsi": True, "qptiff": False}[self.annotationsourcedict[name]],
        position=self.annotationpositiondict[name],
        pscale=self.pscale,
      ) for name in names
    ]
    self.writecsv("annotationinfo", annotationinfos)

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, **kwargs):
    return [
      *super().getoutputfiles(SlideID=SlideID, **kwargs),
      dbloadroot/SlideID/"dbload"/f"{SlideID}_annotationinfo.csv",
    ]

class AnnotationInfoWriterCohortBase(DbloadCohort, AnnotationInfoWriterArgumentParser):
  pass

class WriteAnnotationInfoSample(AnnotationInfoWriterSampleBase, WorkflowSample):
  def run(self):
    xmlfile = self.annotationspolygonsxmlfile
    with open(xmlfile, "rb") as f:
      nodes = jxmlease.parse(f)["Annotations"]["Annotation"]
      if isinstance(nodes, jxmlease.XMLDictNode): nodes = [nodes]
      nodedict = {node.get_xml_attr("Name").lower(): node for node in nodes}
      if len(nodes) != len(nodedict):
        raise ValueError(f"Duplicate annotation names in {xmlfile}: {collections.Counter(node.get_xml_attr('Name') for node in nodes)}")

      self.writeannotationinfo(nodedict.keys())

class WriteAnnotationInfoCohort(AnnotationInfoWriterCohortBase, WorkflowCohort):
  pass

class MergeAnnotationXMLsArgumentParser(AnnotationInfoWriterArgumentParser, DbloadArgumentParser):
  def __init__(self, *args, annotationselectiondict, annotationsourcedict, annotationpositiondict, skipannotations, **kwargs):
    self.__annotationselectiondict = annotationselectiondict
    self.__annotationsourcedict = annotationsourcedict
    self.__annotationpositiondict = annotationpositiondict
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
  def annotationsourcedict(self): return self.__annotationsourcedict
  @property
  def annotationpositiondict(self): return self.__annotationpositiondict
  @property
  def skipannotations(self): return self.__skipannotations

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

class MergeAnnotationXMLsSample(AnnotationInfoWriterSampleBase, WorkflowSample, MergeAnnotationXMLsArgumentParser):
  @property
  def xmloutput(self):
    return self.scanfolder/f"{self.SlideID}_Scan{self.Scan}.annotations.polygons.merged.xml"

  @classmethod
  def getoutputfiles(cls, SlideID, *, im3root, Scan, **kwargs):
    return [
      *super().getoutputfiles(SlideID=SlideID, **kwargs),
      im3root/SlideID/UNIV_CONST.IM3_DIR_NAME/f"Scan{Scan}"/f"{SlideID}_Scan{Scan}.annotations.polygons.merged.xml",
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
          raise ValueError(f"Duplicate annotation names in {xmlfile.name}: {collections.Counter(node.get_xml_attr('Name') for node in nodes)}")
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

    names = [node.get_xml_attr("Name") for node in annotations]
    self.writeannotationinfo(names)

  def run(self, **kwargs):
    return self.mergexmls(**kwargs)

  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return []

class MergeAnnotationXMLsCohort(AnnotationInfoWriterCohortBase, WorkflowCohort, MergeAnnotationXMLsArgumentParser):
  sampleclass = MergeAnnotationXMLsSample
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "annotationselectiondict": self.annotationselectiondict,
      "annotationsourcedict": self.annotationsourcedict,
      "annotationpositiondict": self.annotationpositiondict,
      "skipannotations": self.skipannotations,
    }

def mergeannotationxmlssample(*args, **kwargs):
  MergeAnnotationXMLsSample.runfromargumentparser(*args, **kwargs)

def mergeannotationxmlscohort(*args, **kwargs):
  MergeAnnotationXMLsCohort.runfromargumentparser(*args, **kwargs)

def writeannotationinfosample(*args, **kwargs):
  WriteAnnotationInfoSample.runfromargumentparser(*args, **kwargs)

def writeannotationinfocohort(*args, **kwargs):
  WriteAnnotationInfoCohort.runfromargumentparser(*args, **kwargs)
