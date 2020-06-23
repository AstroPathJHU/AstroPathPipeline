import argparse, contextlib, os, pathlib, re, tempfile, traceback

from ..extractlayer.extractlayer import LayerExtractor, ShredderAndLayerExtractor
from ..utilities import units
from ..baseclasses.sample import SampleDef
from ..utilities.misc import printlogger
from ..utilities.tableio import readtable
from .alignmentset import AlignmentSet

class AlignmentCohort(contextlib.ExitStack):
  def __init__(self, root1, root2, *, doshredding=False, dolayerextraction=False, filter=lambda sample: True, debug=False):
    super().__init__()
    self.root1 = pathlib.Path(root1)
    self.root2 = pathlib.Path(root2) if root2 is not None else root2
    self.doshredding = doshredding
    self.dolayerextraction = dolayerextraction
    self.filter = filter
    self.debug = debug

  def run(self):
    if self.root2 is None:
      raise RuntimeError("If you don't provide the directory of flatw files, you have to run this in a context manager.")

    samples = readtable(self.root1/"sampledef.csv", SampleDef)

    for sample in samples:
      logger = printlogger
      if not sample: continue
      if not self.filter(sample): continue
      try:
        if self.dolayerextraction:
          with (ShredderAndLayerExtractor if self.doshredding else LayerExtractor)(self.root1, self.root2, sample, logger=logger) as extractor:
            logger = extractor.logger
            extractor.extractlayers(alreadyexistsstrategy="skip")

        with AlignmentSet(self.root1, self.root2, sample, uselogfiles=True) as alignmentset:
          logger = alignmentset.logger
          alignmentset.getDAPI()
          alignmentset.align()
          alignmentset.stitch()
      except Exception as e:
        logger.error(str(e).replace(";", ","))
        logger.info(repr(traceback.format_exc()).replace(";", ""))
        if self.debug: raise

class AlignmentCohortTmpDir(AlignmentCohort):
  def __init__(self, root1, *, tmpdirprefix, **kwargs):
    super().__init__(root1, root2=None, dolayerextraction=True, doshredding=True, **kwargs)
    self.tmpdirprefix = tmpdirprefix

  def __enter__(self):
    super().__enter__()
    self.root2 = self.enter_context(tempfile.TemporaryDirectory(prefix=str(self.tmpdirprefix)+os.path.sep))
    return self

  def __exit__(self, *exc):
    super().__exit__(*exc)
    self.root2 = None

  def run(self):
    if self.root2 is None:
      raise RuntimeError("Have to use this in a context manager")
    super().run()

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("root1", type=pathlib.Path)
  g = p.add_mutually_exclusive_group()
  g.add_argument("root2", nargs="?")
  g.add_argument("--tmpprefix", type=pathlib.Path)
  p.add_argument("--sampleregex", type=re.compile)
  p.add_argument("--extractlayer", action="store_true")
  p.add_argument("--shred", action="store_true")
  p.add_argument("--debug", action="store_true")
  p.add_argument("--units", choices=("safe", "fast"), default="fast")
  args = p.parse_args()

  units.setup(args.units)

  kwargs = {"root1": args.root1, "debug": args.debug}
  if args.root2 is not None:
    cls = AlignmentCohort
    kwargs["root2"] = args.root2
    kwargs["doshredding"] = args.shred
    kwargs["dolayerextraction"] = args.extractlayer
  else:
    cls = AlignmentCohortTmpDir
    kwargs["tmpdirprefix"] = args.tmpprefix

  if args.sampleregex is not None:
    kwargs["filter"] = lambda sample: args.sampleregex.match(sample.SlideID)

  with cls(**kwargs) as cohort:
    cohort.run()
