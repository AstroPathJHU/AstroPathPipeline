import gzip, numpy as np, pathlib, PIL.Image
from astropath_calibration.zoom.zoom import Zoom
from .testbase import TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestZoom(TestBaseSaveOutput):
  @classmethod
  def layers(cls, SlideID):
    return {
      "L1_1": (1, 2),
      "M206": (1,),
    }[SlideID]

  @classmethod
  def xys(cls, SlideID):
    return {
      "L1_1": ((0, 0), (0, 1)),
      "M206": ((0, 1), (1, 0)),
    }[SlideID]

  @classmethod
  def selectrectangles(cls, SlideID):
    return {
      "L1_1": (85,),
      "M206": None,
    }[SlideID]

  @property
  def outputfilenames(self):
    return [
      thisfolder/"zoom_test_for_jenkins"/SlideID/"big"/f"{SlideID}-Z9-L{i}-X{x}-Y{y}-big.png"
      for SlideID in ("L1_1", "M206")
      for i in self.layers(SlideID)
      for x, y in self.xys(SlideID)
    ] + [
      thisfolder/"zoom_test_for_jenkins"/SlideID/"wsi"/f"{SlideID}-Z9-L{i}-wsi.png"
      for SlideID in ("L1_1", "M206")
      for i in self.layers(SlideID)
    ]

  def testZoomWsi(self, SlideID="L1_1", keepoutput=False, **kwargs):
    sample = Zoom(thisfolder/"data", SlideID, zoomroot=thisfolder/"zoom_test_for_jenkins", selectrectangles=self.selectrectangles(SlideID), layers=self.layers(SlideID))
    with sample:
      sample.zoom_wsi(**kwargs)

    try:
      for i in self.layers(SlideID):
        for x, y in self.xys(SlideID):
          filename = f"{SlideID}-Z9-L{i}-X{x}-Y{y}-big.png"
          sample.logger.info("comparing "+filename)
          with sample.PILmaximagepixels(), \
               PIL.Image.open(thisfolder/"zoom_test_for_jenkins"/SlideID/"big"/filename) as img, \
               gzip.open(thisfolder/"reference"/"zoom"/SlideID/(filename+".gz")) as refgz, \
               PIL.Image.open(refgz) as targetimg:
            np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))
        filename = f"{SlideID}-Z9-L{i}-wsi.png"
        sample.logger.info("comparing "+filename)
        with sample.PILmaximagepixels(), \
             PIL.Image.open(thisfolder/"zoom_test_for_jenkins"/SlideID/"wsi"/filename) as img, \
             gzip.open(thisfolder/"reference"/"zoom"/SlideID/(filename+".gz")) as refgz, \
             PIL.Image.open(refgz) as targetimg:
          np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))
    except:
      self.saveoutput()
      raise
    else:
      if keepoutput and not (thisfolder/"annowarp_test_for_jenkins"/SlideID).exists():
        (thisfolder/"annowarp_test_for_jenkins").mkdir(exist_ok=True)
        (thisfolder/"zoom_test_for_jenkins"/SlideID).rename(thisfolder/"annowarp_test_for_jenkins"/SlideID)
      self.removeoutput()

  def testZoomWsiFast(self, SlideID="L1_1", **kwargs):
    self.testZoomWsi(SlideID, fast=True, **kwargs)

  def testzoomM206(self, **kwargs):
    self.testZoomWsiFast("M206", keepoutput=True, **kwargs)
