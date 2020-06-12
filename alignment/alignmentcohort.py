import contextlib, dataclasses, pathlib, tempfile
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

class AlignmentCohort:
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

class AlignmentCohortTmpDir(AlignmentCohort, contextlib.ExitStack):
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
