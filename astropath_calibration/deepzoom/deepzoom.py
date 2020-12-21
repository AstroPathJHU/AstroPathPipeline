import numpy as np, os, pathlib, PIL

from ..baseclasses.sample import ZoomSampleBase

class DeepZoomSample(ZoomSampleBase):
  def __init__(self, *args, deepzoomroot, tilesize=256, **kwargs):
    super().__init__(*args, **kwargs)
    self.__deepzoomroot = pathlib.Path(deepzoomroot)
    self.__tilesize = tilesize

  @property
  def logmodule(self): return "deepzoom"

  @property
  def deepzoomroot(self): return self.__deepzoomroot
  @property
  def deepzoomfolder(self): return self.deepzoomroot/self.SlideID

  @property
  def tilesize(self): return self.__tilesize

  def deepzoom_vips(self, layer):
    import pyvips
    self.logger.info("running vips")
    filename = self.wsifilename(layer)
    self.deepzoomfolder.mkdir(parents=True, exist_ok=True)
    dest = self.deepzoomfolder/f"L{layer:d}"
    wsi = pyvips.Image.new_from_file(os.fspath(filename))
    wsi.dzsave(os.fspath(dest), suffix=".png", background=0, depth="onetile", overlap=0, tile_size=self.tilesize)

  def prunezoom(self, layer):
    self.logger.info("checking which files are non-empty")
    destfolder = self.deepzoomfolder/f"L{layer:d}_files"
    minsize = float("inf")
    for nfiles, filename in enumerate(destfolder.glob("*/*.png"), start=1):
      nfiles += 1
      size = filename.stat().st_size
      if size < minsize:
        minsize = size
        fileswithminsize = []
      if size == minsize:
        fileswithminsize.append(filename)

    with PIL.Image.open(fileswithminsize[0]) as im:
      if np.any(im):
        nbad = 0
        del fileswithminsize[:]

    for nbad, filename in enumerate(fileswithminsize, start=1):
      filename.unlink()

    ngood = nfiles - nbad
    self.logger.info("found %d non-empty files out of %d, removing the %d empty ones", ngood, nfiles, nbad)

  def deepzoom(self, layer):
    self.deepzoom_vips(layer)
    self.prunezoom(layer)
