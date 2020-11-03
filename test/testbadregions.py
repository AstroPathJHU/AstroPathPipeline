import numpy as np, os, pathlib, re
from astropathcalibration.badregions.cohort import DustSpeckFinderCohort
from astropathcalibration.badregions.dustspeck import DustSpeckFinder
from astropathcalibration.badregions.sample import DustSpeckFinderSample, TissueFoldFinderSample
from astropathcalibration.badregions.tissuefold import TissueFoldFinderSimple, TissueFoldFinderByCell
from astropathcalibration.utilities import units
from astropathcalibration.utilities.misc import re_subs
from astropathcalibration.utilities.version import astropathversion
from .testbase import TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestBadRegions(TestBaseSaveOutput):
  savedir = thisfolder/"badregions_test_for_jenkins"

  @property
  def outputfilenames(self):
    return [
      thisfolder/"data"/"logfiles"/"dustspeckfinder.log",
      thisfolder/"data"/"M21_1"/"logfiles"/"M21_1-dustspeckfinder.log",
    ]

  @classmethod
  def setUpClass(cls):
    cls.imagesetsmultilayer = []

    f = DustSpeckFinderSample(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1", selectrectangles=[17])
    cls.imagesetsmultilayer.append([r.image for r in f.rectangles])

    f = DustSpeckFinderSample(thisfolder/"data", thisfolder/"data"/"flatw", "M55_1", selectrectangles=[678])
    cls.imagesetsmultilayer.append([f.rectangles[0].image])

    cls.imagesetssinglelayer = []

    f = TissueFoldFinderSample(thisfolder/"data", thisfolder/"data"/"flatw", "M21_1")
    cls.imagesetssinglelayer.append([r.image for r in f.rectangles])

    cls.writeoutreference = False
    try:
      with np.load(thisfolder/"reference"/"badregions"/"badregionsreference.npz") as f:
        cls.reference = dict(f)
    except IOError:
      cls.reference = {}
    cls.seenclasses = set()
    cls.savedir.mkdir(exist_ok=True)

    super().setUpClass()

  @classmethod
  def tearDownClass(cls):
    unknownkeys = set(cls.reference.keys()) - cls.seenclasses
    for _ in unknownkeys:
      cls.writeoutreference = True
      del cls.reference[_]
    if cls.writeoutreference:
      np.savez_compressed(cls.savedir/"badregions.npz", **cls.reference)
    if unknownkeys:
      raise ValueError(f"Unknown arrays in badregionsreference.npz: {', '.join(unknownkeys)}")

    super().tearDownClass()

  def generaltest(self, BRFclass, imagesetindex, imageindex, *, ismultilayer, expectallgood=False, **kwargs):
    brf = BRFclass((self.imagesetsmultilayer if ismultilayer else self.imagesetssinglelayer)[imagesetindex][imageindex])
    badregions = brf.badregions(**kwargs)

    try:
      if expectallgood:
        reference = np.zeros_like(badregions, dtype=badregions.dtype)
      else:
        reference = self.reference[BRFclass.__name__]

        if not np.any(reference):
          raise AssertionError("Reference doesn't have any bad regions, are you sure this is right?  If it is, do expectallgood=True.")
        if BRFclass.__name__ in self.seenclasses:
          raise RuntimeError(f"Trying to run {BRFclass.__name__} on a second image.  Testing is only set up to test each class on one image, unless expectallgood")
        self.seenclasses.add(BRFclass.__name__)

      np.testing.assert_array_equal(badregions, reference)

    except:
      if not expectallgood:
        self.reference[BRFclass.__name__] = badregions
        type(self).writeoutreference = True
        self.seenclasses.add(BRFclass.__name__)
      brf.show(alpha=0, saveas=self.savedir/f"badregions_{BRFclass.__name__}_{imagesetindex}_{imageindex}_image.pdf", **kwargs)
      brf.show(alpha=0.6, saveas=self.savedir/f"badregions_{BRFclass.__name__}_{imagesetindex}_{imageindex}.pdf", **kwargs)
      raise

  def testTissueFoldFinderSimple(self):
    self.generaltest(TissueFoldFinderSimple, 0, 21, threshold=0.15, ismultilayer=False)

  def testTissueFoldFinderByCell(self):
    self.generaltest(TissueFoldFinderByCell, 0, 21, threshold=0.15, ismultilayer=False)

  nodust = []
  for i in range(1):
    def f(self, i=i):
      self.generaltest(DustSpeckFinder, 0, i, expectallgood=True, ismultilayer=True)
    f.__name__ = f"testDustSpeckFinderNoSpeck_{i}"
    nodust.append(f)

  def testDustSpeckFinderWithSpeckFastUnits(self):
    with units.setup_context("fast"):
      self.generaltest(DustSpeckFinder, 1, 0, ismultilayer=True)

  def testCohort(self):
    class TestCohort(DustSpeckFinderCohort):
      def initiatesample(self, *args, **kwargs):
        return super().initiatesample(*args, selectrectangles=[17], **kwargs)
    cohort = TestCohort(thisfolder/"data", thisfolder/"data"/"flatw", debug=True)
    cohort.run()

    for log in (
      thisfolder/"data"/"logfiles"/"dustspeckfinder.log",
      thisfolder/"data"/"M21_1"/"logfiles"/"M21_1-dustspeckfinder.log",
    ):
      ref = thisfolder/"reference"/"badregions"/log.name
      with open(ref) as fref, open(log) as fnew:
        subs = (";[^;]*$", ""),
        refsubs = *subs, (r"(dustspeckfinder )v[\w+.]+", rf"\1{astropathversion}")
        newsubs = *subs,
        refcontents = os.linesep.join([re_subs(line, *refsubs, flags=re.MULTILINE) for line in fref.read().splitlines()])+os.linesep
        newcontents = os.linesep.join([re_subs(line, *newsubs, flags=re.MULTILINE) for line in fnew.read().splitlines()])+os.linesep
        self.assertEqual(refcontents, newcontents)


for f in TestBadRegions.nodust:
  setattr(TestBadRegions, f.__name__, f)
