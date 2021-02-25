import numpy as np, pathlib
from astropath_calibration.utilities import units
from astropath_calibration.zoom.stitchmask import StitchInformMask
from .testbase import TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestStitchMask(TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    return [
      thisfolder/"stitchmask_test_for_jenkins"/SlideID/"{SlideID}-tissuemask.npz"
      for SlideID in ("M206",)
    ]

  def testStitchInformMask(self, SlideID="M206"):
    root = thisfolder/"data"
    maskroot = thisfolder/"stitchmask_test_for_jenkins"
    sample = StitchInformMask(root, SlideID, maskroot=maskroot, logroot=maskroot)
    refsample = StitchInformMask(root, SlideID, maskroot=thisfolder/"reference"/"stitchmask", logroot=maskroot)
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

  def testStitchInformMaskFastUnits(self, SlideID="M206", **kwargs):
    with units.setup_context("fast"):
      self.testStitchInformMask()
