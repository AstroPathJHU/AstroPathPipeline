from ..baseclasses.sample import ZoomSampleBase

class QPTiffAlignmentSample(ZoomSampleBase):
  @contextlib.contextmanager
  def wsi(self, *, layer):
    with PIL.image.open(self.wsifilename(layer=layer)) as wsi: yield wsi
  def align(self):
    with self.wsi(layer=1) as wsi, self.qptiffDAPI() as qptiff:
