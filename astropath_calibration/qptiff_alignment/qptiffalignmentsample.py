import contextlib

from ..baseclasses.sample import ZoomSampleBase

class QPTiffAlignmentSample(ZoomSampleBase):
  @contextlib.contextmanager
  def wsi(self, *, layer):
    with PIL.image.open(self.wsifilename(layer=layer)) as wsi: yield wsi

  @contextlib.contextmanager
  def qptiff(self):
    with QPTiff(self.qptifffilename) as f:
      bestpage = None
      for zoomlevel in f:
        if zoomlevel.imagewidth < 2000:
          break
      yield zoomlevel

  def align(self):
    with self.wsi(layer=1) as wsi, self.qptiff() as qptiff:
      ...
