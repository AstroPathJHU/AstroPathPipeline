import numpy as np, os, pathlib
from astropath.slides.stitchmask.stitchmasksample import StitchAstroPathTissueMaskSample, StitchInformMaskSample
from astropath.slides.stitchmask.stitchmaskcohort import StitchAstroPathTissueMaskCohort, StitchInformMaskCohort
from .testbase import TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestStitchMask(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    for SlideID in "M21_1",:
      oldfolder = thisfolder/"data"/SlideID/"im3"/"meanimage"/"image_masking"
      newfolder = thisfolder/"test_for_jenkins"/"stitchmask"/SlideID/"im3"/"meanimage"/"image_masking"
      for file in oldfolder.glob("*tissue_mask.bin"):
        yield file, newfolder

      newfolder = thisfolder/"test_for_jenkins"/"stitchmask"/SlideID/"dbload"
      for oldfolder in thisfolder/"data"/SlideID/"dbload", thisfolder/"data"/"reference"/"alignment"/SlideID/"dbload":
        for file in oldfolder.glob("*.csv"):
          yield file, newfolder

  @property
  def outputfilenames(self):
    return [
      thisfolder/"test_for_jenkins"/"stitchmask"/SlideID/"im3"/"meanimage"/"image_masking"/f"{SlideID}_{maskfilestem}.{suffix}"
      for SlideID in ("M206", "M21_1")
      for maskfilestem in ("inform_mask", "tissue_mask")
      for suffix in ("npz", "bin")
    ] + [
      thisfolder/"test_for_jenkins"/"stitchmask"/SlideID/"logfiles"/f"{SlideID}-stitch{masktype}mask.log"
      for SlideID in ("M206", "M21_1")
      for masktype in ("inform", "tissue")
    ] + [
      thisfolder/"test_for_jenkins"/"stitchmask"/"logfiles"/f"stitch{masktype}mask.log"
      for masktype in ("inform", "tissue")
    ]

  def _testStitchMask(self, *, SlideID, masktype, maskfilesuffix, units):
    root = thisfolder/"data"
    maskroot = thisfolder/"test_for_jenkins"/"stitchmask"
    samplecls, cohortcls = {
      "inform": (StitchInformMaskSample, StitchInformMaskCohort),
      "astropathtissue": (StitchAstroPathTissueMaskSample, StitchAstroPathTissueMaskCohort),
    }[masktype]

    dbloadroot = None
    if SlideID == "M21_1": dbloadroot = maskroot

    args = [os.fspath(root), "--maskroot", os.fspath(maskroot), "--logroot", os.fspath(maskroot), "--mask-file-suffix", maskfilesuffix, "--allow-local-edits", "--sampleregex", SlideID+"$", "--debug", "--ignore-dependencies", "--rerun-finished", "--units", units, "--use-apiddef", "--project", "0"]
    if dbloadroot is not None: args += ["--dbloadroot", os.fspath(dbloadroot)]
    cohortcls.runfromargumentparser(args)

    sample = samplecls(root, SlideID, maskroot=maskroot, logroot=maskroot, dbloadroot=dbloadroot, maskfilesuffix=maskfilesuffix)
    refsample = samplecls(root, SlideID, maskroot=thisfolder/"data"/"reference"/"stitchmask", logroot=maskroot, dbloadroot=dbloadroot, maskfilesuffix=maskfilesuffix)

    try:
      with sample.using_mask() as mask1, refsample.using_mask() as mask2:
        np.testing.assert_array_equal(mask1, mask2)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testStitchInformMask(self, *, SlideID="M206"):
    self._testStitchMask(SlideID=SlideID, masktype="inform", maskfilesuffix=".npz", units="safe")

  def testStitchInformMaskFastUnits(self, *, SlideID="M206"):
    self._testStitchMask(SlideID=SlideID, masktype="inform", maskfilesuffix=".npz", units="fast")

  def testStitchAstroPathTissueMask(self, SlideID="M21_1"):
    self._testStitchMask(SlideID=SlideID, masktype="astropathtissue", maskfilesuffix=".bin", units="safe")

  def testStitchAstroPathTissueMaskFastUnits(self, SlideID="M21_1"):
    self._testStitchMask(SlideID=SlideID, masktype="astropathtissue", maskfilesuffix=".bin", units="fast")
