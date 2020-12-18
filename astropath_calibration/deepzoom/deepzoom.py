import os, pathlib

from ..baseclasses.sample import ZoomSampleBase

class DeepZoomSample(ZoomSampleBase):
  def __init__(self, *args, deepzoomroot, **kwargs): 
    super().__init__(*args, **kwargs)
    self.__deepzoomroot = pathlib.Path(deepzoomroot)

  @property
  def logmodule(self): return "deepzoom"

  @property
  def deepzoomroot(self): return self.__deepzoomroot
  @property
  def deepzoomfolder(self): return self.deepzoomroot/self.SlideID

  def deepzoom_vips(self, layer):
    import pyvips
    filename = self.wsifilename(layer)
    self.deepzoomfolder.mkdir(parents=True, exist_ok=True)
    dest = self.deepzoomfolder/f"L{layer:d}"
    destfolder = self.deepzoomfolder/f"L{layer:d}_files"
    wsi = pyvips.Image.new_from_file(os.fspath(filename))
    wsi.dzsave(os.fspath(dest), suffix=".png", background=0, depth="onetile", overlap=0, tile_size=256)
