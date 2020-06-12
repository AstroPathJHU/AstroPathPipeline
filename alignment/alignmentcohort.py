import argparse, contextlib, dataclasses, pathlib, re, tempfile
from ..extractlayer.extractlayer import LayerExtractor, ShredderAndLayerExtractor
from ..utilities.tableio import readtable
from .alignmentset import AlignmentSet

@dataclasses.dataclass
class SampleDef:
  SampleID: int
  SlideID: str
  Project: int
  Cohort: int
  Scan: int
  BatchID: int
  isGood: int

  def __bool__(self):
    return bool(self.isGood)

class AlignmentCohort(contextlib.ExitStack):
  def __init__(self, root1, root2, *, doshredding=False, dolayerextraction=False, filter=lambda sample: True):
    self.root1 = pathlib.Path(root1)
    self.root2 = pathlib.Path(root2) if root2 is not None else root2
    self.doshredding = doshredding
    self.dolayerextraction = dolayerextraction
    self.filter = filter

  def run(self):
    if self.root2 is None:
      raise RuntimeError("If you don't provide the directory of flatw files, you have to run this in a context manager.")

    samples = readtable(self.root1/"sampledef.csv", SampleDef)

    for sample in samples:
      if not sample: continue
      if not self.filter(sample): continue
      samp = sample.SlideID
      if self.dolayerextraction:
        extractor = (ShredderAndLayerExtractor if self.doshredding else LayerExtractor)(self.root1, self.root2, samp)
        extractor.extractlayers(alreadyexistsstrategy="skip")
      alignmentset = AlignmentSet(self.root1, self.root2, self.samp, uselogfiles=True)
      try:
        alignmentset.getDAPI()
        alignmentset.align()
        alignmentset.stitch()
      except Exception as e:
        alignmentset.logger.critical("FAILED: "+str(e).replace(",", ""))
        #alignmentset.logger.debug(traceback.format_exc())

class AlignmentCohortTmpDir(AlignmentCohort):
  def __init__(self, root1, *, tmpdirprefix, **kwargs):
    super().__init__(root1, root2=None, dolayerextraction=True, doshredding=True, **kwargs)
    self.tmpdirprefix = tmpdirprefix

  def __enter__(self):
    super().__enter__()
    self.root2 = self.enter_context(tempfile.TemporaryDirectory(prefix=self.tmpdirprefix))

  def __exit__(self, *exc):
    super().__exit__(*exc)
    self.root2 = None

  def run(self):
    if self.root2 is None:
      raise RuntimeError("Have to use this in a context manager")
    super().run()

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("root1")
  p.add_argument("root2", nargs="?")
  p.add_argument("--sampleregex", type=re.compile)
  p.add_argument("--extractlayer", action="store_true")
  p.add_argument("--shred", action="store_true")
  args = p.parse_args()

  kwargs = {"root1": args.root1}
  if args.root2 is not None:
    cls = AlignmentCohort
    kwargs["root2"] = args.root2
    kwargs["doshredding"] = args.shred
    kwargs["dolayerextraction"] = args.extractlayer
  else:
    cls = AlignmentCohortTmpDir

  if args.sampleregex is not None:
    kwargs["filter"] = args.sampleregex.match

  with cls(**kwargs) as cohort:
    cohort.run()
