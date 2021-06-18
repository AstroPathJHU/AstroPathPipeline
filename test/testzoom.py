import gzip, numpy as np, pathlib, PIL.Image
from astropath.slides.zoom.zoomsample import ZoomSample
from astropath.slides.zoom.zoomcohort import ZoomCohort
from .testbase import TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestZoom(TestBaseSaveOutput):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    gunzipreference("M206")
    gunzipreference("L1_1")

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
      thisfolder/"test_for_jenkins"/"zoom"/SlideID/"big"/f"{SlideID}-Z9-L{i}-X{x}-Y{y}-big.png"
      for SlideID in ("L1_1", "M206")
      for i in self.layers(SlideID)
      for x, y in self.xys(SlideID)
    ] + [
      thisfolder/"test_for_jenkins"/"zoom"/SlideID/"wsi"/f"{SlideID}-Z9-L{i}-wsi.png"
      for SlideID in ("L1_1", "M206")
      for i in self.layers(SlideID)
    ]

  def testZoomWsi(self, SlideID="L1_1", units="safe", mode="vips"):
    root = thisfolder/"data"
    zoomroot = thisfolder/"test_for_jenkins"/"zoom"
    args = [str(root), "--zoomroot", str(zoomroot), "--logroot", str(zoomroot), "--sampleregex", SlideID, "--debug", "--units", units, "--mode", mode, "--allow-local-edits", "--ignore-dependencies", "--rerun-finished"]
    if self.selectrectangles(SlideID) is not None:
      args += ["--selectrectangles", *(str(_) for _ in self.selectrectangles(SlideID))]
    if self.layers(SlideID) is not None:
      args += ["--layers", *(str(_) for _ in self.layers(SlideID))]
    ZoomCohort.runfromargumentparser(args)
    sample = ZoomSample(root, SlideID, zoomroot=zoomroot, logroot=zoomroot, selectrectangles=self.selectrectangles(SlideID), layers=self.layers(SlideID))

    try:
      for i in self.layers(SlideID):
        for x, y in self.xys(SlideID):
          filename = f"{SlideID}-Z9-L{i}-X{x}-Y{y}-big.png"
          sample.logger.info("comparing "+filename)
          with sample.PILmaximagepixels(), \
               PIL.Image.open(thisfolder/"test_for_jenkins"/"zoom"/SlideID/"big"/filename) as img, \
               PIL.Image.open(thisfolder/"reference"/"zoom"/SlideID/"big"/filename) as targetimg:
            np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))
        filename = f"{SlideID}-Z9-L{i}-wsi.png"
        sample.logger.info("comparing "+filename)
        with sample.PILmaximagepixels(), \
             PIL.Image.open(thisfolder/"test_for_jenkins"/"zoom"/SlideID/"wsi"/filename) as img, \
             PIL.Image.open(thisfolder/"reference"/"zoom"/SlideID/"wsi"/filename) as targetimg:
          np.testing.assert_array_equal(np.asarray(img), np.asarray(targetimg))
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testZoomWsiFast(self, SlideID="L1_1", **kwargs):
    self.testZoomWsi(SlideID, mode="fast", **kwargs)

  def testzoomM206(self, **kwargs):
    self.testZoomWsi("M206", mode="memmap", **kwargs)

def gunzipreference(SlideID):
  folder = thisfolder/"reference"/"zoom"/SlideID
  for filename in folder.glob("*/*.gz"):
    newfilename = filename.with_suffix("")
    if newfilename.exists(): continue
    with gzip.open(filename) as f, open(newfilename, "wb") as newf:
      newf.write(f.read())
