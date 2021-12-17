import datetime, gzip, more_itertools, numpy as np, os, pathlib, PIL.Image, shutil, tifffile
from astropath.shared.logging import MyLogger
from astropath.utilities.version import astropathversion
from astropath.slides.stitchmask.stitchmasksample import StitchAstroPathTissueMaskSample, StitchInformMaskSample
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
    cls.hackM206mask()

  @classmethod
  def hackM206mask(cls):
    root = thisfolder/"data"
    maskroot = thisfolder/"data"/"reference"/"stitchmask"
    s1 = StitchInformMaskSample(root, "M206", maskroot=maskroot)
    s2 = StitchAstroPathTissueMaskSample(root, "M206")
    filename = s2.maskfilename()
    if not filename.exists():
      with s1.using_tissuemask() as mask:
        np.savez_compressed(s2.maskfilename(), mask=mask)

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
      thisfolder/"test_for_jenkins"/"zoom"/SlideID/"wsi"/f"{SlideID}-Z9-L{i}-wsi.png"
      for SlideID in ("L1_1", "M206")
      for i in self.layers(SlideID)
    ] + [
      thisfolder/"test_for_jenkins"/"zoom"/SlideID/"logfiles"/f"{SlideID}-zoom.log"
      for SlideID in ("L1_1", "M206")
      for i in self.layers(SlideID)
    ] + [
      thisfolder/"test_for_jenkins"/"zoom"/SlideID/"wsi"/f"{SlideID}-Z8-L1-wsi.tiff"
      for SlideID in ("L1_1", "M206")
    ] + [
      thisfolder/"test_for_jenkins"/"zoom"/SlideID/"wsi"/f"{SlideID}-Z8-color-wsi.tiff"
      for SlideID in ("L1_1", "M206")
    ] + [
      thisfolder/"test_for_jenkins"/"zoom"/"logfiles"/"zoom.log"
    ]

  def testZoomWsi(self, SlideID="L1_1", units="safe", mode="vips", tifflayers="color"):
    root = thisfolder/"data"
    zoomroot = thisfolder/"test_for_jenkins"/"zoom"
    selectrectangles = self.selectrectangles(SlideID)
    layers = self.layers(SlideID)
    maskroot = None
    args = [os.fspath(root), "--zoomroot", os.fspath(zoomroot), "--logroot", os.fspath(zoomroot), "--sampleregex", SlideID, "--debug", "--units", units, "--mode", mode, "--allow-local-edits", "--ignore-dependencies"]
    if selectrectangles is not None:
      args += ["--selectrectangles", *(str(_) for _ in selectrectangles)]
    if layers is not None:
      args += ["--layers", *(str(_) for _ in layers)]
    if tifflayers != "color":
      args += ["--tiff-layers", *(str(_) for _ in tifflayers)]
    if maskroot is not None:
      args += ["--maskroot", os.fspath(maskroot)]
    ZoomCohort.runfromargumentparser(args)
    sample = ZoomSample(root, SlideID, zoomroot=zoomroot, logroot=zoomroot, selectrectangles=selectrectangles, layers=layers, tifflayers=tifflayers, maskroot=maskroot)

    try:
      assert not sample.bigfolder.exists()
      tifffilename = sample.wsitifffilename(tifflayers)
      with tifffile.TiffFile(thisfolder/"test_for_jenkins"/"zoom"/SlideID/"wsi"/tifffilename) as tiff, \
           tifffile.TiffFile(thisfolder/"data"/"reference"/"zoom"/SlideID/"wsi"/tifffilename) as targettiff:
        for layer in layers:
          filename = f"{SlideID}-Z9-L{layer}-wsi.png"
          sample.logger.info("comparing "+filename)
          with sample.PILmaximagepixels(), \
               PIL.Image.open(thisfolder/"test_for_jenkins"/"zoom"/SlideID/"wsi"/filename) as img, \
               PIL.Image.open(thisfolder/"data"/"reference"/"zoom"/SlideID/"wsi"/filename) as targetimg:
            imgarray = np.asarray(img)
            targetarray = np.asarray(targetimg)
            np.testing.assert_array_equal(imgarray, targetarray)

        for tiffpage, targettiffpage in more_itertools.zip_equal(tiff.pages, targettiff.pages):
          sample.logger.info("comparing tiff")
          np.testing.assert_array_equal(tiffpage.asarray(), targettiffpage.asarray())

    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testZoomWsiFast(self, SlideID="L1_1", **kwargs):
    self.testZoomWsi(SlideID, mode="fast", **kwargs)

  def testzoomM206(self, **kwargs):
    self.testZoomWsi("M206", mode="memmap", tifflayers=[1], **kwargs)

  def testExistingWSI(self, SlideID="L1_1", **kwargs):
    reffolder = thisfolder/"data"/"reference"/"zoom"/SlideID/"wsi"
    testfolder = thisfolder/"test_for_jenkins"/"zoom"/SlideID/"wsi"
    testfolder.mkdir(exist_ok=True, parents=True)
    for filename in "L1_1-Z9-L1-wsi.png",:
      shutil.copy(reffolder/filename, testfolder)
    logfile = thisfolder/"test_for_jenkins"/"zoom"/SlideID/"logfiles"/f"{SlideID}-zoom.log"
    logfile.parent.mkdir(exist_ok=True, parents=True)
    with open(logfile, "w", newline="\r\n") as f:
      now = datetime.datetime.now().strftime(MyLogger.dateformat)
      f.write(f"0;0;L1_1;START: zoom {astropathversion};{now}\n")
      f.write(f"0;0;L1_1;FINISH: zoom {astropathversion};{now}\n")
    self.testZoomWsi(SlideID=SlideID, **kwargs)

  def testExistingWsiFast(self, SlideID="L1_1", **kwargs):
    self.testExistingWsi(SlideID, mode="fast", **kwargs)

def gunzipreference(SlideID):
  folder = thisfolder/"data"/"reference"/"zoom"/SlideID
  for filename in folder.glob("*/*.gz"):
    newfilename = filename.with_suffix("")
    if newfilename.exists(): continue
    with gzip.open(filename) as f, open(newfilename, "wb") as newf:
      newf.write(f.read())
