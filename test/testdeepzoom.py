import more_itertools, numpy as np, os, pathlib, PIL.Image
from astropath.slides.deepzoom.deepzoomsample import DeepZoomFile, DeepZoomSample
from astropath.slides.deepzoom.deepzoomcohort import DeepZoomCohort
from astropath.utilities.optionalimports import pyvips
from astropath.utilities.tableio import readtable
from .testbase import assertAlmostEqual, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestDeepZoom(TestBaseSaveOutput):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    from .testzoom import gunzipreference
    gunzipreference("M206")

  def testDeepZoom(self, SlideID="M206", units="safe", empty=False, **kwargs):
    root = thisfolder/"data"
    zoomroot = thisfolder/"data"/"reference"/"zoom"
    deepzoomroot = thisfolder/"test_for_jenkins"/"deepzoom"
    refroot = thisfolder/"data"/"reference"/"deepzoom"
    if empty:
      deepzoomroot /= "empty"
      refroot /= "empty"
      oldfilename = zoomroot/SlideID/"wsi"/f"{SlideID}-Z9-L1-wsi.png"
      zoomroot = thisfolder/"test_for_jenkins"/"deepzoom"/"empty"/"zoom"
      newfilename = zoomroot/SlideID/"wsi"/f"{SlideID}-Z9-L1-wsi.png"
      newfilename.parent.mkdir(parents=True, exist_ok=True)
      blank = pyvips.Image.new_from_file(os.fspath(oldfilename)).linear(0, 0)
      blank.pngsave(os.fspath(newfilename))

    args = [str(root), "--zoomroot", str(zoomroot), "--deepzoomroot", str(deepzoomroot), "--logroot", str(deepzoomroot), "--sampleregex", SlideID, "--debug", "--units", units, "--layers", "1", "--allow-local-edits", "--ignore-dependencies", "--rerun-finished"]
    DeepZoomCohort.runfromargumentparser(args)

    sample = DeepZoomSample(root, SlideID, zoomroot=zoomroot, deepzoomroot=deepzoomroot, logroot=deepzoomroot)
    zoomlist = sample.deepzoomfolder/"zoomlist.csv"

    try:
      folder = sample.deepzoomfolder/"L1_files"
      reffolder = refroot/SlideID/"L1_files"
      for filename, reffilename in more_itertools.zip_equal(
        sorted(folder.glob("*/*.png")),
        sorted(reffolder.glob("*/*.png")),
      ):
        basename = filename.name
        refbasename = reffilename.name
        assert basename <= refbasename, f"{refbasename} exists in reference, but was not created in the test"
        assert basename >= refbasename, f"{basename} was created in the test, but does not exist in reference"
        sample.logger.info(f"comparing {basename}")
        with PIL.Image.open(filename) as im, PIL.Image.open(reffilename) as ref:
          np.testing.assert_array_equal(np.asarray(im), np.asarray(ref))

        new = readtable(zoomlist, DeepZoomFile, checkorder=True, checknewlines=True)
        ref = readtable(refroot/SlideID/zoomlist.name, DeepZoomFile, checkorder=True, checknewlines=True)
        for resultnew, resultref in more_itertools.zip_equal(new, ref):
          assertAlmostEqual(resultnew, resultref, rtol=0, atol=0)

    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testEmptyDeepZoom(self, **kwargs):
    return self.testDeepZoom(empty=True, **kwargs)

  @property
  def outputfilenames(self):
    for folder in (
      thisfolder/"test_for_jenkins"/"deepzoom",
      thisfolder/"test_for_jenkins"/"deepzoom"/"empty",
    ):
      reffolder = thisfolder/"data"/"reference"/folder.relative_to(thisfolder/"test_for_jenkins")
      yield folder/"logfiles"/"deepzoom.log"
      for SlideID in "M206",:
        yield folder/SlideID/"L1.dzi"
        yield folder/SlideID/"zoomlist.csv"
        yield folder/SlideID/"logfiles"/f"{SlideID}-deepzoom.log"
        for filename in (reffolder/SlideID/"L1_files").glob("*/*.png"):
          yield folder/filename.relative_to(reffolder)

  @property
  def deletefilenames(self):
    return [
      thisfolder/"test_for_jenkins"/"deepzoom"/"empty"/"zoom"/SlideID/"wsi"/f"{SlideID}-Z9-L1-wsi.png"
      for SlideID in ("M206",)
    ]
