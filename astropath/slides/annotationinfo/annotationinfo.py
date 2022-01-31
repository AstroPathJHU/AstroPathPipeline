import collections, methodtools, numpy as np, re
from ...shared.annotationpolygonxmlreader import AllowedAnnotation, XMLPolygonAnnotationFile, XMLPolygonAnnotationFileInfoWriter
from ...shared.argumentparser import DbloadArgumentParser, XMLPolygonFileArgumentParser
from ...shared.cohort import DbloadCohort, XMLPolygonFileCohort, WorkflowCohort
from ...shared.csvclasses import AnnotationInfo
from ...shared.sample import DbloadSample, WorkflowSample, XMLPolygonAnnotationFileSample
from ...utilities import units
from ...utilities.misc import ArgParseAddRegexToDict, ArgParseAddToDict
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

class WriteAnnotationInfoSample(ReadAffineShiftSample, XMLPolygonAnnotationFileSample, WorkflowSample, AnnotationInfoWriterArgumentParser, XMLPolygonAnnotationFileInfoWriter):
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
    annotationsxmlregex = kwargs.get("annotationsxmlregex", None)
    im3root = kwargs["im3root"]
    Scan = kwargs["Scan"]
    SlideID = kwargs["SlideID"]
    xmls = [
      _ for _ in (im3root/SlideID/"im3"/f"Scan{Scan}").glob(f"*{SlideID}*annotations.polygons*.xml")
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

class WriteAnnotationInfoCohort(DbloadCohort, XMLPolygonFileCohort, WorkflowCohort, AnnotationInfoWriterArgumentParser, XMLPolygonFileArgumentParser):
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

class CopyAnnotationInfoArgumentParserBase(DbloadArgumentParser):
  def __init__(self, *args, renameannotations, **kwargs):
    self.__renameannotations = renameannotations
    [AllowedAnnotation.allowedannotation(v) for k, v in renameannotations.items()]
    super().__init__(*args, **kwargs)

  @property
  def renameannotations(self): return self.__renameannotations

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--rename-annotation", nargs=2, action=ArgParseAddToDict, metavar=("XMLNAME", "NEWNAME"), dest="renameannotations", help="Rename an annotation given in the xml file to a new name (which has to be in the master list)")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "renameannotations": parsed_args_dict.pop("renameannotations"),
    }

class CopyAnnotationInfoArgumentParser(CopyAnnotationInfoArgumentParserBase, XMLPolygonFileArgumentParser):
  pass

class CopyAnnotationInfoSampleBase(DbloadSample, WorkflowSample, CopyAnnotationInfoArgumentParserBase):
  @classmethod
  def getoutputfiles(cls, **kwargs):
    SlideID = kwargs["SlideID"]
    dbloadroot = kwargs["dbloadroot"]
    return [
      *super().getoutputfiles(**kwargs),
      dbloadroot/SlideID/"dbload"/f"{SlideID}_annotationinfo.csv",
    ]

  @classmethod
  def logmodule(cls):
    return "copyannotationinfo"

  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return [
      *super().workflowdependencyclasses(**kwargs),
      WriteAnnotationInfoSample,
    ]

  @property
  def outlineannotationinfo(self):
    return AnnotationInfo(
      sampleid=self.SampleID,
      originalname="outline",
      dbname="outline",
      annotationsource="mask",
      position=None,
      pscale=self.pscale,
      apscale=self.apscale,
      xmlfile=None,
      xmlsha=None,
      scanfolder=self.scanfolder,
    )

  def renameannotationinfos(self, infos):
    dct = {info.originalname: info for info in infos}
    for oldname, newname in self.renameannotations.items():
      try:
        info = dct[oldname]
      except KeyError:
        raise ValueError(f"Trying to rename annotation {oldname}, which doesn't exist")
      info.dbname = newname
    ctr = collections.Counter(info.dbname for info in infos)
    if max(ctr.values()) > 1:
      raise ValueError(f"Multiple annotations with the same name after renaming: {ctr}")

class CopyAnnotationInfoSample(CopyAnnotationInfoSampleBase, XMLPolygonAnnotationFileSample, WorkflowSample, CopyAnnotationInfoArgumentParser):
  def run(self):
    annotationinfos = self.readtable(self.annotationinfofile, AnnotationInfo)
    annotationinfos.append(self.outlineannotationinfo)
    self.renameannotationinfos(annotationinfos)
    self.writecsv("annotationinfo", annotationinfos)

  def inputfiles(self, **kwargs):
    result = [
      *super().inputfiles(**kwargs),
      self.annotationinfofile,
    ]
    return result

class CopyAnnotationInfoCohortBase(DbloadCohort, WorkflowCohort, CopyAnnotationInfoArgumentParserBase):
  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "renameannotations": self.renameannotations,
    }

class CopyAnnotationInfoCohort(CopyAnnotationInfoCohortBase, DbloadCohort, XMLPolygonFileCohort, WorkflowCohort, CopyAnnotationInfoArgumentParser):
  sampleclass = CopyAnnotationInfoSample

class MergeAnnotationXMLsArgumentParser(CopyAnnotationInfoArgumentParserBase):
  def __init__(self, *args, annotationselectiondict, skipannotations, **kwargs):
    self.__annotationselectiondict = annotationselectiondict
    self.__skipannotations = skipannotations
    super().__init__(*args, **kwargs)

    duplicates = frozenset(self.annotationselectiondict) & frozenset(self.skipannotations)
    if duplicates:
      raise ValueError(f"Some annotations are specified to be taken from an xml file and to be skipped: {', '.join(duplicates)}")

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

class MergeAnnotationXMLsSample(CopyAnnotationInfoSampleBase, MergeAnnotationXMLsArgumentParser):
  @methodtools.lru_cache()
  @property
  def allxmls(self):
    return frozenset(
      filename
      for filename in self.scanfolder.glob(f"*{self.SlideID}*annotations.polygons*.xml")
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
    info = {}
    allnames = set()
    for xmlfile in self.allxmls:
      xmlfile = XMLPolygonAnnotationFile(xmlfile=xmlfile, pscale=self.pscale, apscale=self.apscale)
      infodict = {info.originalname: info for info in xmlfile.annotationinfo}
      allnames.update(infodict.keys())
      info[xmlfile.annotationspolygonsxmlfile] = infodict

    unknown = allnames - set(self.annotationxmldict) - set(self.skipannotations)
    if unknown:
      raise ValueError(f"Found unknown annotations in xml files: {', '.join(sorted(unknown))}")

    mergedinfo = []
    for name, xmlfile in self.annotationxmldict.items():
      self.logger.info(f"Taking {name} from {xmlfile.name}")
      mergedinfo.append(info[xmlfile][name.lower()])

    for name in sorted(set(self.skipannotations) & allnames):
      self.logger.info(f"Skipping {name}")

    mergedinfo.append(self.outlineannotationinfo)

    self.renameannotationinfos(mergedinfo)

    self.writecsv("annotationinfo", mergedinfo)

  def run(self, **kwargs):
    return self.mergexmls(**kwargs)

  def inputfiles(self, **kwargs):
    xmls = [
      _ for _ in self.scanfolder.glob(f"*{self.SlideID}*annotations.polygons*.xml")
    ]
    return [
      *super().inputfiles(**kwargs),
      *(xml.with_suffix(".annotationinfo.csv") for xml in xmls)
    ]

class MergeAnnotationXMLsCohort(CopyAnnotationInfoCohortBase, DbloadCohort, WorkflowCohort, MergeAnnotationXMLsArgumentParser):
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

def writeannotationinfosample(*args, **kwargs):
  WriteAnnotationInfoSample.runfromargumentparser(*args, **kwargs)

def writeannotationinfocohort(*args, **kwargs):
  WriteAnnotationInfoCohort.runfromargumentparser(*args, **kwargs)

def copyannotationinfosample(*args, **kwargs):
  CopyAnnotationInfoSample.runfromargumentparser(*args, **kwargs)

def copyannotationinfocohort(*args, **kwargs):
  CopyAnnotationInfoCohort.runfromargumentparser(*args, **kwargs)

def mergeannotationxmlssample(*args, **kwargs):
  MergeAnnotationXMLsSample.runfromargumentparser(*args, **kwargs)

def mergeannotationxmlscohort(*args, **kwargs):
  MergeAnnotationXMLsCohort.runfromargumentparser(*args, **kwargs)
