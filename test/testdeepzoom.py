import more_itertools, numpy as np, pathlib, PIL.Image
from astropath.slides.deepzoom.deepzoomsample import DeepZoomFile, DeepZoomSample
from astropath.slides.deepzoom.deepzoomcohort import DeepZoomCohort
from astropath.utilities.tableio import readtable
from .testbase import assertAlmostEqual, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestDeepZoom(TestBaseSaveOutput):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    from .testzoom import gunzipreference
    gunzipreference("M206")

  def testDeepZoom(self, SlideID="M206", units="safe", **kwargs):
    root = thisfolder/"data"
    zoomroot = thisfolder/"reference"/"zoom"
    deepzoomroot = thisfolder/"deepzoom_test_for_jenkins"
    args = [str(root), "--zoomroot", str(zoomroot), "--deepzoomroot", str(deepzoomroot), "--logroot", str(deepzoomroot), "--sampleregex", SlideID, "--debug", "--units", units, "--layers", "1", "--allow-local-edits", "--ignore-dependencies", "--rerun-finished"]
    DeepZoomCohort.runfromargumentparser(args)

    sample = DeepZoomSample(root, SlideID, zoomroot=zoomroot, deepzoomroot=deepzoomroot, logroot=deepzoomroot)
    zoomlist = sample.deepzoomfolder/"zoomlist.csv"

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

        new = readtable(zoomlist, DeepZoomFile)
        ref = readtable(thisfolder/"reference"/"deepzoom"/SlideID/zoomlist.name, DeepZoomFile)
        for resultnew, resultref in more_itertools.zip_equal(new, ref):
          resultnew.name = pathlib.PurePosixPath(resultnew.name.relative_to(thisfolder))
          resultref.name = pathlib.PurePosixPath(resultref.name.relative_to(resultref.name.parent.parent.parent.parent.parent))
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
