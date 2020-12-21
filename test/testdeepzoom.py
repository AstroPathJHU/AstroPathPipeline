import gzip, more_itertools, numpy as np, pathlib, PIL.Image
from astropath_calibration.deepzoom.deepzoom import DeepZoomSample
from .testbase import TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestDeepZoom(TestBaseSaveOutput):
  def testDeepZoom(self, SlideID="M206", **kwargs):
    sample = DeepZoomSample(thisfolder/"data", SlideID, zoomroot=thisfolder/"annowarp_test_for_jenkins", deepzoomroot=thisfolder/"deepzoom_test_for_jenkins")
    with sample:
      sample.deepzoom(layer=1, **kwargs)

    try:
      folder = sample.deepzoomfolder/"L1_files"
      reffolder = thisfolder/"reference"/"deepzoom"/SlideID/"L1_files"
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
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  @property
  def outputfilenames(self):
    return sum(
      (
        [
          thisfolder/"deepzoom_test_for_jenkins"/SlideID/"L1_files"/filename.parent.name/filename.name
          for filename in (thisfolder/"reference"/"deepzoom"/SlideID/"L1_files").glob("*/*.png")
        ] + [thisfolder/"deepzoom_test_for_jenkins"/SlideID/"L1.dzi"]
        for SlideID in ("M206",)
      ), []
    )
