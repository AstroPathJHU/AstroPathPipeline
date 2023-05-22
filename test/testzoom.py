import datetime, gzip, job_lock, more_itertools, numpy as np, os, pathlib, PIL.Image, shutil, tifffile
from astropath.shared.astropath_logging import MyLogger
from astropath.utilities.version import astropathversion
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
    from .data.M206.im3.meanimage.image_masking.hackmask import hackmask
    hackmask()

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

  def testZoomWsi(self, SlideID="L1_1", units="safe", mode="vips", tifflayers="color", usesample=False):
    root = thisfolder/"data"
    zoomroot = thisfolder/"test_for_jenkins"/"zoom"
    selectrectangles = self.selectrectangles(SlideID)
    layers = self.layers(SlideID)
    maskroot = None
    args = [os.fspath(root), "--zoomroot", os.fspath(zoomroot), "--logroot", os.fspath(zoomroot), "--units", units, "--mode", mode, "--allow-local-edits"]
    if usesample:
      args += [SlideID]
    else:
      args += ["--sampleregex", SlideID, "--ignore-dependencies", "--debug"]
    if selectrectangles is not None:
      args += ["--selectrectangles", *(str(_) for _ in selectrectangles)]
    if layers is not None:
      args += ["--layers", *(str(_) for _ in layers)]
    if tifflayers is None:
      pass
    elif tifflayers == "color":
      args += ["--tiff-layers", tifflayers]
    else:
      args += ["--tiff-layers", *(f"{_:d}" for _ in tifflayers)]
    if maskroot is not None:
      args += ["--maskroot", os.fspath(maskroot)]
    (ZoomSample if usesample else ZoomCohort).runfromargumentparser(args)
    sample = ZoomSample(root, SlideID, zoomroot=zoomroot, logroot=zoomroot, selectrectangles=selectrectangles, layers=layers, tifflayers=tifflayers, maskroot=maskroot)

    try:
      assert not sample.bigfolder.exists()
      for layer in layers:
        filename = f"{SlideID}-Z9-L{layer}-wsi.png"
        sample.logger.info("comparing "+filename)
        with sample.PILmaximagepixels(), \
             PIL.Image.open(thisfolder/"test_for_jenkins"/"zoom"/SlideID/"wsi"/filename) as img, \
             PIL.Image.open(thisfolder/"data"/"reference"/"zoom"/SlideID/"wsi"/filename) as targetimg:
          imgarray = np.asarray(img)
          targetarray = np.asarray(targetimg)
          np.testing.assert_allclose(imgarray, targetarray, atol=1)

      if tifflayers is not None:
        tifffilename = sample.wsitifffilename(tifflayers)
        with tifffile.TiffFile(thisfolder/"test_for_jenkins"/"zoom"/SlideID/"wsi"/tifffilename) as tiff, \
             tifffile.TiffFile(thisfolder/"data"/"reference"/"zoom"/SlideID/"wsi"/tifffilename) as targettiff:
          for tiffpage, targettiffpage in more_itertools.zip_equal(tiff.pages, targettiff.pages):
            sample.logger.info("comparing tiff")
            np.testing.assert_array_equal(tiffpage.asarray(), targettiffpage.asarray())

    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testZoomWsiFast(self, SlideID="L1_1", **kwargs):
    self.testZoomWsi(SlideID, mode="fast", tifflayers=None, **kwargs)

  def testzoomM206(self, **kwargs):
    self.testZoomWsi("M206", mode="memmap", tifflayers=[1], **kwargs)

  def testExistingWSI(self, SlideID="L1_1", mode="vips", **kwargs):
    reffolder = thisfolder/"data"/"reference"/"zoom"/SlideID/"wsi"
    testfolder = thisfolder/"test_for_jenkins"/"zoom"/SlideID/"wsi"
    testfolder.mkdir(exist_ok=True, parents=True)
    for filename, corrupt in (
      ("L1_1-Z9-L1-wsi.png", False),
      ("L1_1-Z9-L2-wsi.png", True),
    ):
      with open(reffolder/filename, "rb") as f, open(testfolder/filename, "wb") as newf:
        shutil.copyfileobj(f, newf)
        if corrupt:
          newf.seek(-1, os.SEEK_END)
          newf.truncate()

    if mode == "vips":
      bigreffolder = thisfolder/"data"/"reference"/"zoom"/SlideID/"big"
      bigfolder = thisfolder/"test_for_jenkins"/"zoom"/SlideID/"big"
      bigfolder.mkdir(exist_ok=True, parents=True)
      X0Y0file = bigfolder/f"{SlideID}-Z9-L2-X0-Y0-big.tiff"
      X0Y1file = bigfolder/f"{SlideID}-Z9-L2-X0-Y1-big.tiff"
      bigreffile = bigreffolder/X0Y0file.name
      for filename, corrupt in (
        (X0Y0file, False),
        (X0Y1file, True),
      ):
        with open(bigreffile, "rb") as f, open(filename, "wb") as newf:
          shutil.copyfileobj(f, newf)
          if corrupt:
            newf.seek(-1, os.SEEK_END)
            newf.truncate()
    elif mode in ("fast", "memmap"):
      pass
    else:
      assert False, mode

    logfile = thisfolder/"test_for_jenkins"/"zoom"/SlideID/"logfiles"/f"{SlideID}-zoom.log"
    logfile.parent.mkdir(exist_ok=True, parents=True)
    with open(logfile, "w", newline="\r\n") as f:
      now = datetime.datetime.now().strftime(MyLogger.dateformat)
      f.write(f"0;0;L1_1;START: zoom-test {astropathversion};{now}\n")
      f.write(f"0;0;L1_1;FINISH: zoom {astropathversion};{now}\n")
    self.testZoomWsi(SlideID=SlideID, mode=mode, **kwargs)

  def testExistingWSIFast(self, SlideID="L1_1", **kwargs):
    self.testExistingWSI(SlideID, mode="fast", tifflayers=None, usesample=True, **kwargs)

def gunzipreference(SlideID):
  folder = thisfolder/"data"/"reference"/"zoom"/SlideID
  for filename in folder.glob("*/*.gz"):
    with job_lock.JobLockAndWait(filename.with_suffix(".lock"), task=f"gunzipping {filename}", delay=10):
      newfilename = filename.with_suffix("")
      if newfilename.exists(): continue
      with gzip.open(filename) as f, open(newfilename, "wb") as newf:
        newf.write(f.read())
