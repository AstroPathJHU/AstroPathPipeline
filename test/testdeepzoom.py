import more_itertools, numpy as np, pathlib, PIL.Image
from astropath_calibration.deepzoom.deepzoom import DeepZoomFile, DeepZoomSample
from astropath_calibration.utilities.tableio import readtable
from .testbase import assertAlmostEqual, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestDeepZoom(TestBaseSaveOutput):
  def testDeepZoom(self, SlideID="M206", **kwargs):
    sample = DeepZoomSample(thisfolder/"data", SlideID, zoomroot=thisfolder/"annowarp_test_for_jenkins", deepzoomroot=thisfolder/"deepzoom_test_for_jenkins", layers=[1])
    with sample:
      sample.deepzoom(**kwargs)

    zoomlist = sample.csv("zoomlist")
    movedzoomlist = sample.deepzoomfolder/zoomlist.name
    zoomlist.rename(movedzoomlist)

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

        new = readtable(movedzoomlist, DeepZoomFile)
        ref = readtable(thisfolder/"reference"/"deepzoom"/SlideID/movedzoomlist.name, DeepZoomFile)
        for resultnew, resultref in more_itertools.zip_equal(new, ref):
          resultnew.fname = pathlib.PurePosixPath(resultnew.fname.relative_to(thisfolder))
          resultref.fname = pathlib.PurePosixPath(resultref.fname.relative_to(resultref.fname.parent.parent.parent.parent.parent))
          assertAlmostEqual(resultnew, resultref, rtol=0, atol=0)

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
        ] + [
          thisfolder/"deepzoom_test_for_jenkins"/SlideID/"L1.dzi",
          thisfolder/"deepzoom_test_for_jenkins"/SlideID/f"{SlideID}_zoomlist.csv",
        ]
        for SlideID in ("M206",)
      ), []
    )
