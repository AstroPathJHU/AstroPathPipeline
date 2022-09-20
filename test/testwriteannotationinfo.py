import os, pathlib
from astropath.shared.csvclasses import AnnotationInfo
from astropath.slides.annotationinfo.annotationinfo import CopyAnnotationInfoCohort, CopyAnnotationInfoSample, MergeAnnotationXMLsCohort, MergeAnnotationXMLsSample, WriteAnnotationInfoCohort, WriteAnnotationInfoSample
from .testbase import compare_two_csv_files, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestWriteAnnotationInfo(TestBaseCopyInput, TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    for root in (
      thisfolder/"test_for_jenkins"/"writeannotationinfo",
      thisfolder/"test_for_jenkins"/"writeannotationinfo"/"emptyannotation",
    ):
      yield root/"M206"/"dbload"/"M206_annotationinfo.csv",

      yield root/"M206"/"im3"/"Scan1"/"M206_Scan1.annotations.polygons.annotationinfo.csv",
      yield root/"M206"/"im3"/"Scan1"/"M206_Scan1.annotations.polygons_2.annotationinfo.csv",

      yield root/"logfiles"/"copyannotationinfo.log",
      yield root/"logfiles"/"writeannotationinfo.log",

      yield root/"M206"/"logfiles"/"M206-copyannotationinfo.log",
      yield root/"M206"/"logfiles"/"M206-writeannotationinfo.log",
  @classmethod
  def filestocopy(cls):
    sourceroot = thisfolder/"data"
    destroot = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    destrootempty = thisfolder/"test_for_jenkins"/"writeannotationinfo"/"emptyannotation"
    yield sourceroot/"upkeep_and_progress"/"AstropathAPIDdef_0.csv", destroot/"upkeep_and_progress"
    for SlideID in "M206",:
      yield sourceroot/SlideID/"im3"/"Scan1"/f"{SlideID}_Scan1.annotations.polygons.xml", destroot/SlideID/"im3"/"Scan1"
      yield sourceroot/SlideID/"im3"/"Scan1"/f"{SlideID}_Scan1.annotations.polygons.xml", destroot/SlideID/"im3"/"Scan1", f"{SlideID}_Scan1.annotations.polygons_2.xml"
      yield sourceroot/SlideID/"im3"/"Scan1"/f"{SlideID}_Scan1.annotations.polygons.xml", destrootempty/SlideID/"im3"/"Scan1", None, r"Tumor(.*<Regions>).*(</Regions>)", r"Tumour\1\2"
      for csv in "constants", "affine":
        yield sourceroot/SlideID/"dbload"/f"{SlideID}_{csv}.csv", destroot/SlideID/"dbload"
        yield sourceroot/SlideID/"dbload"/f"{SlideID}_{csv}.csv", destrootempty/SlideID/"dbload"


  def testWriteAnnotationInfo(self, *, SlideID="M206", units="safe", variant=None, usecohort=True):
    root = im3root = dbloadroot = logroot = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    refroot = thisfolder/"data"/"reference"/"writeannotationinfo"
    if variant is None:
      pass
    elif variant == "empty":
      im3root = im3root/"emptyannotation"
      refroot = refroot/"emptyannotation"
    else:
      raise ValueError(variant)
    regex = ".*annotations.polygons.xml"
    args = [os.fspath(root), "--im3root", os.fspath(im3root), "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(logroot), "--annotations-on-qptiff", "--units", units, "--allow-local-edits", "--annotations-xml-regex", regex, "--project", "0"]
    s = WriteAnnotationInfoSample(root=root, dbloadroot=dbloadroot, im3root=im3root, logroot=logroot, samp=SlideID, annotationsource="qptiff", annotationposition=None, annotationpositionfromaffineshift=False, annotationsxmlregex=regex)

    try:
      if usecohort:
        WriteAnnotationInfoCohort.runfromargumentparser(args + ["--ignore-dependencies", "--sampleregex", SlideID, "--debug", "--use-apiddef"])
      else:
        WriteAnnotationInfoSample.runfromargumentparser(args + [SlideID])

      new = s.annotationinfofile
      reffolder = refroot/SlideID/"im3"/f"Scan{s.Scan}"
      extrakwargs = {_: getattr(s, _) for _ in ("pscale", "apscale", "scanfolder")}
      compare_two_csv_files(new.parent, reffolder, new.name, AnnotationInfo, extrakwargs=extrakwargs)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testWriteAnnotationInfoFastUnits(self, **kwargs):
    self.testWriteAnnotationInfo(units="fast_pixels", **kwargs)

  def testWriteAnnotationInfoSample(self, **kwargs):
    self.testWriteAnnotationInfo(usecohort=False, **kwargs)

  def testCopyAnnotationInfo(self, SlideID="M206", units="safe", variant=None):
    root = thisfolder/"data"
    dbloadroot = logroot = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    im3root = root/"reference"/"writeannotationinfo"
    refroot = root/"reference"/"writeannotationinfo"
    if variant is None:
      pass
    elif variant == "empty":
      dbloadroot /= "emptyannotation"
      logroot /= "emptyannotation"
      im3root /= "emptyannotation"
      refroot /= "emptyannotation"
    else:
      raise ValueError(variant)
    args = [os.fspath(root), "--im3root", os.fspath(im3root), "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(logroot), "--sampleregex", SlideID, "--debug", "--units", units, "--ignore-dependencies", "--allow-local-edits"]
    s = CopyAnnotationInfoSample(root=root, dbloadroot=dbloadroot, im3root=im3root, logroot=logroot, samp=SlideID, renameannotations={})

    try:
      CopyAnnotationInfoCohort.runfromargumentparser(args)

      new = s.csv("annotationinfo")
      reffolder = refroot/SlideID/"dbload"
      extrakwargs = {_: getattr(s, _) for _ in ("pscale", "apscale", "scanfolder")}
      compare_two_csv_files(new.parent, reffolder, new.name, AnnotationInfo, extrakwargs=extrakwargs)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testCopyAnnotationInfoFastUnits(self, **kwargs):
    self.testCopyAnnotationInfo(units="fast_microns", **kwargs)

  def testSkipAnnotation(self, *, SlideID="M206", units="safe"):
    root = thisfolder/"data"
    dbloadroot = im3root = logroot = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    s = MergeAnnotationXMLsSample(root=root, im3root=im3root, dbloadroot=dbloadroot, samp=SlideID, annotationselectiondict={}, skipannotations=set(), renameannotations={})
    commonargs = [os.fspath(root), "--im3root", os.fspath(im3root), "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(logroot), "--sampleregex", SlideID, "--debug", "--units", units, "--ignore-dependencies", "--allow-local-edits"]
    write1args = ["--annotations-on-qptiff", "--annotations-xml-regex", ".*annotations.polygons.xml"]
    write2args = ["--annotations-on-qptiff", "--annotations-xml-regex", ".*annotations.polygons_2.xml"]
    mergeargs = ["--annotation", "Good tissue", ".*annotations.polygons.xml", "--skip-annotation", "tumor"]

    try:
      WriteAnnotationInfoCohort.runfromargumentparser(commonargs+write1args)
      WriteAnnotationInfoCohort.runfromargumentparser(commonargs+write2args)
      MergeAnnotationXMLsCohort.runfromargumentparser(commonargs+mergeargs)

      new = s.csv("annotationinfo")
      reffolder = root/"reference"/"writeannotationinfo"/"skipannotation"/SlideID/"dbload"
      extrakwargs = {_: getattr(s, _) for _ in ("pscale", "apscale", "scanfolder")}
      compare_two_csv_files(new.parent, reffolder, new.name, AnnotationInfo, extrakwargs=extrakwargs)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testSkipAnnotationFastUnits(self, **kwargs):
    self.testSkipAnnotation(units="fast_microns", **kwargs)

  def testAnnotationPosition(self, *, SlideID="M206", units="safe"):
    root = thisfolder/"data"
    dbloadroot = im3root = logroot = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    s = MergeAnnotationXMLsSample(root=root, im3root=im3root, dbloadroot=dbloadroot, samp=SlideID, annotationselectiondict={}, skipannotations=set(), renameannotations={})
    commonargs = [os.fspath(root), "--im3root", os.fspath(im3root), "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(logroot), "--sampleregex", SlideID, "--debug", "--units", units, "--ignore-dependencies", "--allow-local-edits"]
    regex1 = ".*annotations.polygons.xml"
    regex2 = ".*annotations.polygons_2.xml"
    write1args = ["--annotations-on-wsi", "--annotation-position", "100", "100", "--annotations-xml-regex", regex1]
    write2args = ["--annotations-on-wsi", "--annotation-position-from-affine-shift", "--annotations-xml-regex", regex2]
    mergeargs = ["--annotation", "Good tissue", regex1, "--annotation", "tumor", regex2]

    try:
      WriteAnnotationInfoCohort.runfromargumentparser(commonargs+write1args)
      WriteAnnotationInfoCohort.runfromargumentparser(commonargs+write2args)

      MergeAnnotationXMLsCohort.runfromargumentparser(commonargs+mergeargs)

      new = s.csv("annotationinfo")
      reffolder = root/"reference"/"writeannotationinfo"/"annotationposition"/SlideID/"dbload"
      extrakwargs = {_: getattr(s, _) for _ in ("pscale", "apscale", "scanfolder")}
      compare_two_csv_files(new.parent, reffolder, new.name, AnnotationInfo, extrakwargs=extrakwargs)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testAnnotationPositionFastUnits(self, **kwargs):
    self.testAnnotationPosition(units="fast_microns", **kwargs)

  def testRenameAnnotation(self, *, SlideID="M206", units="safe"):
    root = thisfolder/"data"
    dbloadroot = im3root = logroot = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    s = MergeAnnotationXMLsSample(root=root, im3root=im3root, dbloadroot=dbloadroot, samp=SlideID, annotationselectiondict={}, skipannotations=set(), renameannotations={})
    commonargs = [os.fspath(root), "--im3root", os.fspath(im3root), "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(logroot), "--sampleregex", SlideID, "--debug", "--units", units, "--ignore-dependencies", "--allow-local-edits"]
    write1args = ["--annotations-on-qptiff", "--annotations-xml-regex", ".*annotations.polygons.xml"]
    write2args = ["--annotations-on-qptiff", "--annotations-xml-regex", ".*annotations.polygons_2.xml"]
    mergeargs = ["--annotation", "Good tissue", ".*annotations.polygons.xml", "--annotation", "tumor", ".*annotations.polygons_2.xml", "--rename-annotation", "outline", "good tissue", "--rename-annotation", "good tissue", "good tissue x", "--rename-annotation", "tumor", "tumor x"]

    try:
      WriteAnnotationInfoCohort.runfromargumentparser(commonargs+write1args)
      WriteAnnotationInfoCohort.runfromargumentparser(commonargs+write2args)
      MergeAnnotationXMLsCohort.runfromargumentparser(commonargs+mergeargs)

      new = s.csv("annotationinfo")
      reffolder = root/"reference"/"writeannotationinfo"/"renameannotation"/SlideID/"dbload"
      extrakwargs = {_: getattr(s, _) for _ in ("pscale", "apscale", "scanfolder")}
      compare_two_csv_files(new.parent, reffolder, new.name, AnnotationInfo, extrakwargs=extrakwargs)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testWriteEmptyAnnotation(self, **kwargs):
    self.testWriteAnnotationInfo(variant="empty", **kwargs)
  def testCopyEmptyAnnotation(self, **kwargs):
    self.testCopyAnnotationInfo(variant="empty", **kwargs)
