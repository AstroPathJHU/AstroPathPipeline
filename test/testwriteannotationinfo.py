import jxmlease, os, pathlib
from astropath.shared.csvclasses import AnnotationInfo
from astropath.slides.annowarp.mergeannotationxmls import MergeAnnotationXMLsCohort, MergeAnnotationXMLsSample
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
    yield thisfolder/"data"/"M206"/"im3"/"Scan1"/"M206_Scan1.annotations.polygons.xml", thisfolder/"test_for_jenkins"/"writeannotationinfo"/"M206"/"im3"/"Scan1"
    yield thisfolder/"data"/"M206"/"dbload"/"M206_constants.csv", thisfolder/"test_for_jenkins"/"writeannotationinfo"/"M206"/"dbload"

  def testMergeAnnotationXMLs(self):
    root = thisfolder/"data"
    dbloadroot = im3root = thisfolder/"test_for_jenkins"/"writeannotationinfo"
    SlideID = "M206"
    args = [os.fspath(root), "--im3root", os.fspath(im3root), "--dbloadroot", os.fspath(dbloadroot), "--sampleregex", SlideID, "--annotation", "good tissue", ".*[.]xml", "--skip-annotation", "tumor", "--debug", "--no-log", "--annotations-on-qptiff"]
    s = MergeAnnotationXMLsSample(root=root, im3root=im3root, dbloadroot=im3root, samp=SlideID, annotationselectiondict={}, annotationsourcedict={}, annotationpositiondict={}, skipannotations=set())

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
