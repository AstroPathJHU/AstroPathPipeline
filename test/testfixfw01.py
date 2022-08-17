import numpy as np, os, pathlib, shutil
from astropath.hpfs.fixfw01.fixfw01cohort import FixFW01CohortDbload, FixFW01CohortXML
from astropath.hpfs.fixfw01.fixfw01sample import FixFW01SampleDbload, FixFW01SampleXML
from astropath.utilities.miscfileio import memmapcontext
from astropath.utilities.miscmath import floattoint
from .testbase import TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestFixFW01(TestBaseCopyInput, TestBaseSaveOutput):
  hpfsdict = {
    "M21_1": ["M21_1_[45093,13253]"],
  }

  @property
  def outputfilenames(self):
    for SlideID, hpfs in self.hpfsdict.items():
      for hpf in hpfs:
        yield thisfolder/"test_for_jenkins"/"fixfw01"/"flatw"/SlideID/f"{hpf}.fw01"
      yield thisfolder/"test_for_jenkins"/"fixfw01"/SlideID/"logfiles"/f"{SlideID}-fixfw01.log"
    yield thisfolder/"test_for_jenkins"/"fixfw01"/"logfiles"/"fixfw01.log"

  @classmethod
  def filestocopy(cls):
    oldflatw = thisfolder/"data"/"flatw"
    newflatw = thisfolder/"test_for_jenkins"/"fixfw01"/"flatw"
    oldroot = thisfolder/"data"
    newroot = thisfolder/"test_for_jenkins"/"fixfw01"

    for SlideID, hpfs in cls.hpfsdict.items():
      for hpf in hpfs:
        yield oldflatw/SlideID/f"{hpf}.fw", newflatw/SlideID
      yield oldroot/SlideID/"dbload"/f"{SlideID}_rect.csv", newroot/SlideID/"dbload"

  def testFixFW01(self, dbload=False, SlideID="M21_1", corruptaction="--error-on-corrupt", removeoutput=True):
    root = thisfolder/"data"
    logroot = dbloadroot = thisfolder/"test_for_jenkins"/"fixfw01"
    shardedim3root = thisfolder/"test_for_jenkins"/"fixfw01"/"flatw"
    selectrectangles = {
      "M21_1": [17],
    }[SlideID]
    args = [
      os.fspath(root),
      "--shardedim3root", os.fspath(shardedim3root),
      "--logroot", os.fspath(logroot),
      "--sampleregex", f"^{SlideID}$",
      "--selectrectangles", *(str(_) for _ in selectrectangles),
      corruptaction,
      "--allow-local-edits",
      "--debug",
      "--ignore-dependencies",
    ]
    if dbload:
      args += [
        "--dbloadroot", os.fspath(dbloadroot),
      ]
    cohortclass = FixFW01CohortDbload if dbload else FixFW01CohortXML
    sampleclass = FixFW01SampleDbload if dbload else FixFW01SampleXML
    try:
      cohortclass.runfromargumentparser(args)

      kwargs = {"root": root, "shardedim3root": shardedim3root, "logroot": logroot, "samp": SlideID, "selectrectangles": selectrectangles}
      if dbload: kwargs["dbloadroot"] = dbloadroot
      samp = sampleclass(**kwargs)

      for filename in samp.outputfiles:
        reffilename = thisfolder/"data"/"flatw"/SlideID/filename.name
        r = samp.rectangles[0]
        shape = tuple(floattoint((r.imageshape / r.onepixel).astype(float))[::-1])
        with \
          memmapcontext(filename, dtype=np.uint16, shape=shape, order="F", mode="r") as data, \
          memmapcontext(reffilename, dtype=np.uint16, shape=shape, order="F", mode="r") as ref \
        :
          np.testing.assert_array_equal(data, ref)

    except:
      if removeoutput: self.saveoutput()
      raise
    finally:
      if removeoutput: self.removeoutput()

  def testFixFW01Dbload(self, **kwargs):
    self.testFixFW01(dbload=True, **kwargs)

  def testExists(self, **kwargs):
    for src, dest in self.filestocopy():
      if src.suffix == ".fw":
        shutil.copy(src.with_suffix(".fw01"), dest)
    self.testFixFW01(**kwargs)

  def testEmpty(self, **kwargs):
    for src, dest in self.filestocopy():
      if src.suffix == ".fw":
        (dest/src.with_suffix(".fw01").name).touch()
    with self.assertRaises(IOError):
      self.testFixFW01(corruptaction="--error-on-corrupt", removeoutput=False, **kwargs)
    self.testFixFW01(corruptaction="--remove-corrupt", removeoutput=False, **kwargs)
    self.testFixFW01(corruptaction="--remove-disagreement", **kwargs)

  def testWrong(self, **kwargs):
    for src, dest in self.filestocopy():
      if src.suffix == ".fw":
        filename = shutil.copy(src.with_suffix(".fw01"), dest)
        with memmapcontext(filename, dtype=np.uint16, shape=1, order="F", mode="r+") as data:
          data[0] += 1

    with self.assertRaises(ValueError):
      self.testFixFW01(corruptaction="--error-on-corrupt", removeoutput=False, **kwargs)
    with self.assertRaises(ValueError):
      self.testFixFW01(corruptaction="--remove-corrupt", removeoutput=False, **kwargs)
    self.testFixFW01(corruptaction="--remove-disagreement", **kwargs)
