import more_itertools, pathlib

from astropath.samples.crossregistration.crossregistration import CrossRegAffineMatrix, CrossRegAlignmentResult, CrossRegistration

from .testbase import assertAlmostEqual, TestBaseCopyInput, TestBaseSaveOutput
from .testzoom import gunzipreference as gunzipzoom

thisfolder = pathlib.Path(__file__).parent

class TestCrossRegistration(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    gunzipzoom("M21_1")
    gunzipzoom("MA12")

  @classmethod
  def filestocopy(cls):
    for SlideID in "M21_1", "MA12":
      newdbload = thisfolder/"test_for_jenkins"/"crossregistration"/SlideID/"dbload"
      for olddbload in thisfolder/"data"/SlideID/"dbload", thisfolder/"reference"/"alignment"/"M21_1"/"dbload":
        for csv in (
          "constants",
          "fields",
        ):
          filename = olddbload/f"{SlideID}_{csv}.csv"
          if filename.exists():
            yield filename, newdbload

  @property
  def outputfilenames(self):
    return [
      thisfolder/"test_for_jenkins"/"crossregistration"/SlideID/"dbload"/f"{SlideID}_{csv}.csv"
      for csv in ("xwarp", "xform")
      for SlideID in ("M21_1", "MA12")
    ] + [
      thisfolder/"test_for_jenkins"/"crossregistration"/SlideID/"logfiles"/f"{SlideID}-{log}.log"
      for log in ("crossregistration",)
      for SlideID in ("M21_1", "MA12")
    ] + [
      thisfolder/"test_for_jenkins"/"crossregistration"/"logfiles"/f"{log}.log"
      for log in ("crossregistration",)
    ]

  def testCrossRegistration(self, units="safe"):
    root = thisfolder/"data"
    testroot = thisfolder/"test_for_jenkins"/"crossregistration"
    zoomroot = root/"reference"/"zoom"
    maskroot1 = root/"reference"/"stitchmask"
    samp1 = "M21_1"
    samp2 = "MA12"
    #args = [os.fspath(root), os.fspath(root), "--dbloadroot", os.fspath(testroot), os.fspath(testroot), "--logroot", os.fspath(testroot), os.fspath(testroot), "--zoomroot", os.fspath(zoomroot), os.fspath(zoomroot), "--maskroot1", os.fspath(maskroot), os.fspath(root), "--units", units, "--sampleregex", f"^({samp1}|{samp2})$", "--debug", "--allow-local-edits", "--ignore-dependencies"]

    r = CrossRegistration(root1=root, root2=root, samp1=samp1, samp2=samp2, dbloadroot1=testroot, dbloadroot2=testroot, logroot1=testroot, logroot2=testroot, zoomroot1=zoomroot, zoomroot2=zoomroot, maskroot1=maskroot1, uselogfiles=False)
    r.runalignment()

    #CrossRegistration.runfromargumentparser(args=args)

    reffolder = thisfolder/"data"/"reference"/"crossregistration"/"M21_1-MA12"/"dbload"
    try:
      for s in r.samples:
        for csv, rowclass in (
          ("xform", CrossRegAffineMatrix),
          ("xwarp", CrossRegAlignmentResult),
        ):
          try:
            filename = s.csv(csv)
            reffilename = reffolder/filename.name.replace(s.SlideID, "M21_1-MA12")
            rows = s.readtable(filename, rowclass, checkorder=True, checknewlines=True)
            targetrows = s.readtable(reffilename, rowclass, checkorder=True, checknewlines=True)
            for row, target in more_itertools.zip_equal(rows, targetrows):
              assertAlmostEqual(row, target, rtol=1e-5)
          except:
            raise ValueError(f"Error in {csv}")
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()
