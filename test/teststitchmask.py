import numpy as np, pathlib
from astropath.utilities import units
from astropath.slides.zoom.stitchmasksample import StitchAstroPathTissueMaskSample, StitchInformMaskSample
from .testbase import TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestStitchMask(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    for SlideID in "M21_1",:
      oldfolder = thisfolder/"data"/SlideID/"im3"/"meanimage"/"image_masking"
      newfolder = thisfolder/"stitchmask_test_for_jenkins"/SlideID/"im3"/"meanimage"/"image_masking"
      for file in oldfolder.glob("*tissue_mask.bin"):
        yield file, newfolder
  @property
  def outputfilenames(self):
    return [
      thisfolder/"stitchmask_test_for_jenkins"/SlideID/"im3"/"meanimage"/"image_masking"/f"{SlideID}_{maskfilestem}.{suffix}"
      for SlideID in ("M206", "M21_1")
      for maskfilestem in ("inform_mask", "tissue_mask")
      for suffix in ("npz", "bin")
    ]

  def _testStitchMask(self, *, SlideID, masktype, maskfilesuffix):
    root = thisfolder/"data"
    maskroot = thisfolder/"stitchmask_test_for_jenkins"
    samplecls = {
      "inform": StitchInformMaskSample,
      "astropathtissue": StitchAstroPathTissueMaskSample,
    }[masktype]
    dbloadroot = None
    if SlideID == "M21_1": dbloadroot = thisfolder/"reference"/"alignment"
    sample = samplecls(root, SlideID, maskroot=maskroot, logroot=maskroot, dbloadroot=dbloadroot, maskfilesuffix=maskfilesuffix)
    refsample = samplecls(root, SlideID, maskroot=thisfolder/"reference"/"stitchmask", logroot=maskroot, dbloadroot=dbloadroot)
    sample.writemask()

    try:
      mask1 = sample.readmask()
      mask2 = refsample.readmask()
      np.testing.assert_array_equal(mask1, mask2)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testStitchInformMask(self, *, SlideID="M206"):
    self._testStitchMask(SlideID=SlideID, masktype="inform", maskfilesuffix=".npz")

  def testStitchAstroPathTissueMaskFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self._testStitchMask(SlideID=SlideID, masktype="astropathtissue", maskfilesuffix=".bin")
