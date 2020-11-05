import gzip, numpy as np, pathlib, PIL.Image
from astropath_calibration.zoom.zoom import Zoom
from astropath_calibration.utilities.misc import PILmaximagepixels
from .testbase import TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestZoom(TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    return [
      thisfolder/"zoom_test_for_jenkins"/SlideID/"big"/f"{SlideID}-Z9-L{i}-X0-Y{y}-big.png"
      for SlideID in ("L1_1",)
      for i in range(1, 3)
      for y in {
        "L1_1": (0, 1)
      }[SlideID]
    ] + [
      thisfolder/"zoom_test_for_jenkins"/SlideID/"wsi"/f"{SlideID}-Z9-L{i}-wsi.png"
      for SlideID in ("L1_1",)
      for i in range(1, 3)
    ]

  def testZoomWsi(self, SlideID="L1_1", **kwargs):
    sample = Zoom(thisfolder/"data", thisfolder/"flatw", SlideID, zoomroot=thisfolder/"zoom_test_for_jenkins", selectrectangles=[85], layers=(1, 2))
    with sample:
      sample.zoom_wsi(**kwargs)

    try:
      for i in range(1, 3):
        for y in {
          "L1_1": (0, 1)
        }[SlideID]:
          filename = f"{SlideID}-Z9-L{i}-X0-Y{y}-big.png"
          sample.logger.info("comparing "+filename)
          with PILmaximagepixels(sample.tilesize**2), \
               PIL.Image.open(thisfolder/"zoom_test_for_jenkins"/SlideID/"big"/filename) as img, \
               gzip.open(thisfolder/"reference"/"zoom"/SlideID/(filename+".gz")) as refgz, \
               PIL.Image.open(refgz) as targetimg:
            np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))
        filename = f"{SlideID}-Z9-L{i}-wsi.png"
        sample.logger.info("comparing "+filename)
        with PILmaximagepixels(np.product(sample.ntiles)*sample.tilesize**2), \
             PIL.Image.open(thisfolder/"zoom_test_for_jenkins"/SlideID/"wsi"/filename) as img, \
             gzip.open(thisfolder/"reference"/"zoom"/SlideID/(filename+".gz")) as refgz, \
             PIL.Image.open(refgz) as targetimg:
          np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testZoomWsiFast(self, SlideID="L1_1"):
    self.testZoomWsi(SlideID, fast=True)
