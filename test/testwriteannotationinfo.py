import jxmlease, os, pathlib, unittest
from astropath.shared.csvclasses import AnnotationInfo
from astropath.slides.annowarp.mergeannotationxmls import MergeAnnotationXMLsCohort, MergeAnnotationXMLsSample, WriteAnnotationInfoCohort, WriteAnnotationInfoSample
from .testbase import compare_two_csv_files, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestWriteAnnotationInfo(TestBaseCopyInput, TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    return [
      thisfolder/"test_for_jenkins"/"writeannotationinfo"/"M206"/"im3"/"Scan1"/"M206_Scan1.annotations.polygons.merged.xml",
      thisfolder/"test_for_jenkins"/"writeannotationinfo"/"M206"/"im3"/"Scan1"/"M206_Scan1.annotations.polygons.annotationinfo.csv",
      thisfolder/"test_for_jenkins"/"writeannotationinfo"/"M206"/"logfiles"/"M206-writeannotationinfo.log",
    ]
  @classmethod
  def filestocopy(cls):
    for SlideID in "M206",:
      yield thisfolder/"data"/SlideID/"im3"/"Scan1"/f"{SlideID}_Scan1.annotations.polygons.xml", thisfolder/"test_for_jenkins"/"writeannotationinfo"/SlideID/"im3"/"Scan1"
      for csv in "constants", "affine":
        yield thisfolder/"data"/SlideID/"dbload"/f"{SlideID}_{csv}.csv", thisfolder/"test_for_jenkins"/"writeannotationinfo"/SlideID/"dbload"

  def testWriteAnnotationInfo(self, *, SlideID="M206", units="safe"):
    root = thisfolder/"data"
    im3root = dbloadroot = logroot = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    args = [os.fspath(root), "--im3root", os.fspath(im3root), "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(logroot), "--sampleregex", SlideID, "--debug", "--annotations-on-qptiff", "--units", units, "--ignore-dependencies", "--allow-local-edits"]
    s = WriteAnnotationInfoSample(root=root, dbloadroot=dbloadroot, im3root=im3root, logroot=logroot, samp=SlideID, annotationsource="qptiff", annotationposition=None, annotationpositionfromaffineshift=False)

    try:
      WriteAnnotationInfoCohort.runfromargumentparser(args)

      new = s.annotationinfofilename
      reffolder = root/"reference"/"writeannotationinfo"/SlideID/"im3"/f"Scan{s.Scan}"
      extrakwargs = {_: getattr(s, _) for _ in ("pscale",)}
      compare_two_csv_files(new.parent, reffolder, new.name, AnnotationInfo, extrakwargs=extrakwargs)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testWriteAnnotationInfoFastUnits(self, **kwargs):
    self.testWriteAnnotationInfo(units="fast_pixels", **kwargs)

  def testSkipAnnotation(self, *, SlideID="M206", units="safe"):
    root = thisfolder/"data"
    dbloadroot = im3root = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    args = [os.fspath(root), "--im3root", os.fspath(im3root), "--dbloadroot", os.fspath(dbloadroot), "--sampleregex", SlideID, "--annotation", "Good tissue", ".*[.]xml", "--skip-annotation", "tumor", "--debug", "--no-log", "--annotations-on-qptiff", "--units", units, "--ignore-dependencies"]
    s = MergeAnnotationXMLsSample(root=root, im3root=im3root, dbloadroot=dbloadroot, samp=SlideID, annotationselectiondict={}, annotationsourcedict={}, annotationpositiondict={}, skipannotations=set())

    try:
      MergeAnnotationXMLsCohort.runfromargumentparser(args)

      new = s.csv("annotationinfo")
      reffolder = root/"reference"/"writeannotationinfo"/"skipannotation"/SlideID/"dbload"
      extrakwargs = {_: getattr(s, _) for _ in ("pscale",)}
      compare_two_csv_files(new.parent, reffolder, new.name, AnnotationInfo, extrakwargs=extrakwargs)

      with open(im3root/SlideID/"im3"/"Scan1"/"M206_Scan1.annotations.polygons.merged.xml", "rb") as f:
        newxml = jxmlease.parse(f)
      with open(root/SlideID/"im3"/"Scan1"/"M206_Scan1.annotations.polygons.xml", "rb") as f:
        oldxml = jxmlease.parse(f)
      del oldxml["Annotations"]["Annotation"][1]
      oldxml["Annotations"]["Annotation"], = oldxml["Annotations"]["Annotation"]
      self.assertEqual(newxml, oldxml)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testSkipAnnotationFastUnits(self, **kwargs):
    self.testSkipAnnotation(units="fast_microns", **kwargs)

  def testAnnotationPosition(self, *, SlideID="M206", units="safe"):
    root = thisfolder/"data"
    dbloadroot = im3root = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    args = [os.fspath(root), "--im3root", os.fspath(im3root), "--dbloadroot", os.fspath(dbloadroot), "--sampleregex", SlideID, "--annotation", "Good tissue", ".*[.]xml", "--annotation", "tumor", ".*[.]xml", "--debug", "--no-log", "--annotation-source", "tumor", "wsi", "--annotation-source", "Good tissue", "wsi", "--annotation-position", "Good tissue", "100", "100", "--units", units, "--ignore-dependencies"]
    s = MergeAnnotationXMLsSample(root=root, im3root=im3root, dbloadroot=dbloadroot, samp=SlideID, annotationselectiondict={}, annotationsourcedict={}, annotationpositiondict={}, skipannotations=set())

    try:
      MergeAnnotationXMLsCohort.runfromargumentparser(args)

      new = s.csv("annotationinfo")
      reffolder = root/"reference"/"writeannotationinfo"/"annotationposition"/SlideID/"dbload"
      extrakwargs = {_: getattr(s, _) for _ in ("pscale",)}
      compare_two_csv_files(new.parent, reffolder, new.name, AnnotationInfo, extrakwargs=extrakwargs)

      with open(im3root/SlideID/"im3"/"Scan1"/"M206_Scan1.annotations.polygons.merged.xml", "rb") as f:
        newxml = jxmlease.parse(f)
      with open(root/SlideID/"im3"/"Scan1"/"M206_Scan1.annotations.polygons.xml", "rb") as f:
        oldxml = jxmlease.parse(f)
      self.assertEqual(newxml, oldxml)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testAnnotationPositionFastUnits(self, **kwargs):
    self.testAnnotationPosition(units="fast_microns", **kwargs)
