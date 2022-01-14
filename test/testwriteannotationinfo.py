import jxmlease, os, pathlib
from astropath.shared.csvclasses import AnnotationInfo
from astropath.slides.annowarp.mergeannotationxmls import MergeAnnotationXMLsCohort, MergeAnnotationXMLsSample, WriteAnnotationInfoCohort, WriteAnnotationInfoSample
from .testbase import compare_two_csv_files, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestWriteAnnotationInfo(TestBaseCopyInput, TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    return [
      thisfolder/"test_for_jenkins"/"writeannotationinfo"/"M206"/"im3"/"Scan1"/"M206_Scan1.annotations.polygons.merged.xml",
      thisfolder/"test_for_jenkins"/"writeannotationinfo"/"M206"/"dbload"/"M206_annotationinfo.csv",
    ]
  @classmethod
  def filestocopy(cls):
    for SlideID in "M206",:
      yield thisfolder/"data"/SlideID/"im3"/"Scan1"/f"{SlideID}_Scan1.annotations.polygons.xml", thisfolder/"test_for_jenkins"/"writeannotationinfo"/SlideID/"im3"/"Scan1"
      for csv in "constants", "affine":
        yield thisfolder/"data"/SlideID/"dbload"/f"{SlideID}_{csv}.csv", thisfolder/"test_for_jenkins"/"writeannotationinfo"/SlideID/"dbload"

  def testWriteAnnotationInfo(self, *, SlideID="M206", units="safe"):
    root = thisfolder/"data"
    dbloadroot = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    args = [os.fspath(root), "--dbloadroot", os.fspath(dbloadroot), "--sampleregex", SlideID, "--debug", "--no-log", "--annotations-on-qptiff", "--units", units, "--ignore-dependencies"]
    s = WriteAnnotationInfoSample(root=root, dbloadroot=dbloadroot, samp=SlideID, annotationsourcedict={}, annotationpositiondict={})

    try:
      WriteAnnotationInfoCohort.runfromargumentparser(args)

      new = s.csv("annotationinfo")
      reffolder = root/"reference"/"writeannotationinfo"/"mergeannotationxmls"/SlideID/"dbload"
      extrakwargs = {_: getattr(s, _) for _ in ("pscale",)}
      compare_two_csv_files(new.parent, reffolder, new.name, AnnotationInfo, extrakwargs=extrakwargs)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testWriteAnnotationInfoFastUnits(self, **kwargs):
    self.testWriteAnnotationInfo(units="fast_pixels", **kwargs)

  def testMergeAnnotationXMLs(self, *, SlideID="M206", units="safe"):
    root = thisfolder/"data"
    dbloadroot = im3root = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    args = [os.fspath(root), "--im3root", os.fspath(im3root), "--dbloadroot", os.fspath(dbloadroot), "--sampleregex", SlideID, "--annotation", "good tissue", ".*[.]xml", "--skip-annotation", "tumor", "--debug", "--no-log", "--annotations-on-qptiff", "--units", units, "--ignore-dependencies"]
    s = MergeAnnotationXMLsSample(root=root, im3root=im3root, dbloadroot=dbloadroot, samp=SlideID, annotationselectiondict={}, annotationsourcedict={}, annotationpositiondict={}, skipannotations=set())

    try:
      MergeAnnotationXMLsCohort.runfromargumentparser(args)

      new = s.csv("annotationinfo")
      reffolder = root/"reference"/"writeannotationinfo"/"mergeannotationxmls"/SlideID/"dbload"
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

  def testMergeAnnotationXMLsFastUnits(self, **kwargs):
    self.testMergeAnnotationXMLs(units="fast_microns", **kwargs)
