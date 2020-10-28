import gzip, numpy as np, pathlib, PIL.Image
from ..zoom.zoom import Zoom
from ..utilities import units
from ..utilities.misc import PILmaximagepixels
from .testbase import TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestZoom(TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    return [
      thisfolder/"zoom_test_for_jenkins"/SlideID/"big"/f"{SlideID}-Z9-L{i}-X0-Y0-big.png"
      for SlideID in ("M21_1",)
      for i in range(1, 9)
    ] + [
      thisfolder/"zoom_test_for_jenkins"/SlideID/"wsi"/f"{SlideID}-Z9-L{i}-wsi.png"
      for SlideID in ("M21_1",)
      for i in range(1, 9)
    ]

  def testZoomWsi(self, SlideID="M21_1", **kwargs):
    sample = Zoom(thisfolder/"data", thisfolder/"flatw", SlideID, zoomroot=thisfolder/"zoom_test_for_jenkins", uselogfiles=True, selectrectangles=[17])
    with sample:
      sample.zoom_wsi(**kwargs)

    try:
      for i in range(1, 9):
        with PILmaximagepixels(sample.tilesize**2), \
             PIL.Image.open(thisfolder/"zoom_test_for_jenkins"/SlideID/"big"/f"{SlideID}-Z9-L{i}-X0-Y0-big.png") as img, \
             gzip.open(thisfolder/"reference"/"zoom"/SlideID/f"{SlideID}-Z9-L{i}-X0-Y0-big.png.gz") as refgz, \
             PIL.Image.open(refgz) as targetimg:
          np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))
        with PILmaximagepixels(sample.tilesize**2), \
             PIL.Image.open(thisfolder/"zoom_test_for_jenkins"/SlideID/"wsi"/f"{SlideID}-Z9-L{i}-wsi.png") as img, \
             gzip.open(thisfolder/"reference"/"zoom"/SlideID/f"{SlideID}-Z9-L{i}-wsi.png.gz") as refgz, \
             PIL.Image.open(refgz) as targetimg:
          np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testZoomFast(self, SlideID="M21_1"):
    with units.setup_context("fast"):
      self.testZoom(SlideID, fast=True)
