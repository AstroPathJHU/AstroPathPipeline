import collections, dataclasses, functools, numpy as np, os, pathlib, PIL, re

from ..baseclasses.sample import DbloadSampleBase, DeepZoomSampleBase, ReadRectanglesComponentTiff, ZoomSampleBase
from ..utilities.tableio import pathfield, writetable

class DeepZoomSample(ReadRectanglesComponentTiff, DbloadSampleBase, ZoomSampleBase, DeepZoomSampleBase):
  def __init__(self, *args, tilesize=256, **kwargs):
    super().__init__(*args, **kwargs)
    self.__tilesize = tilesize

  @property
  def logmodule(self): return "deepzoom"

  @property
  def tilesize(self): return self.__tilesize

  def layerfolder(self, layer): return self.deepzoomfolder/f"L{layer:d}_files"

  def deepzoom_vips(self, layer):
    import pyvips
    self.logger.info("running vips for layer %d", layer)
    filename = self.wsifilename(layer)
    self.deepzoomfolder.mkdir(parents=True, exist_ok=True)
    destfolder = self.layerfolder(layer)
    if destfolder.exists():
      for subfolder in destfolder.iterdir():
        if subfolder.is_dir(): subfolder.rmdir()
      destfolder.rmdir()
    dest = destfolder.with_name(destfolder.name.replace("_files", ""))
    wsi = pyvips.Image.new_from_file(os.fspath(filename))
    wsi.dzsave(os.fspath(dest), suffix=".png", background=0, depth="onetile", overlap=0, tile_size=self.tilesize)

  def prunezoom(self, layer):
    self.logger.info("checking which files are non-empty for layer %d", layer)
    destfolder = self.layerfolder(layer)
    filesizedict = collections.defaultdict(list)
    for nfiles, filename in enumerate(destfolder.glob("*/*.png"), start=1):
      size = filename.stat().st_size
      filesizedict[size].append(filename)

    nbad = 0

    for size, files in sorted(filesizedict.items()):
      with PIL.Image.open(files[0]) as im:
        if np.any(im):
          if im.size == (self.tilesize, self.tilesize):
            break
          else:
            continue

      self.logger.info("removing %d empty files with file size %d", len(files), size)
      for nbad, filename in enumerate(files, start=nbad+1):
        filename.unlink()

    ngood = nfiles - nbad
    self.logger.info("there are %d remaining non-empty files", ngood)

  def patchzoom(self, layer):
    self.logger.info("relabeling zooms for layer %d", layer)
    destfolder = self.layerfolder(layer)
    folders = sorted(destfolder.iterdir(), key=lambda x: int(x.name))
    maxfolder = int(folders[-1].name)
    if maxfolder > 9:
      raise ValueError(f"Need more zoom levels than 0-9 (max from vips is {maxfolder})")
    for folder in reversed(folders):
      newnumber = int(folder.name) + 9 - maxfolder
      newfolder = destfolder/f"Z{newnumber}"
      folder.rename(newfolder)

    minzoomnumber = newnumber
    minzoomfolder = newfolder

    smallestimagefilename = minzoomfolder/"0_0.png"
    with PIL.Image.open(smallestimagefilename) as im:
      im.load()
    n, m = im.size
    if m > self.tilesize or n > self.tilesize:
      raise ValueError(f"{smallestimagefilename} is too big {m}x{n}, expected <= {self.tilesize}x{self.tilesize}")
    if m < self.tilesize or n < self.tilesize:
      im = PIL.Image.fromarray(np.pad(np.asarray(im), ((0, self.tilesize-m), (0, self.tilesize-n))))
      im.save(smallestimagefilename)

    smallestimage = im

    for i in range(minzoomnumber):
      newfolder = destfolder/f"Z{i}"
      newfolder.mkdir(exist_ok=True)
      newfilename = newfolder/"0_0.png"
      im = smallestimage.resize(np.asarray(smallestimage.size) // 2**(minzoomnumber-i))
      im = np.asarray(im)
      im = (im * 1.25**(minzoomnumber-i)).astype(np.uint8)
      m, n = im.shape
      im = np.pad(im, ((0, self.tilesize-m), (0, self.tilesize-n)))
      im = PIL.Image.fromarray(im)
      im.save(newfilename)

  def writezoomlist(self):
    lst = []
    for layer in self.layers:
      folder = self.layerfolder(layer)
      for zoomfolder in sorted(folder.iterdir()):
        zoom = int(re.match("Z([0-9]*)", zoomfolder.name).group(1))
        for filename in sorted(zoomfolder.iterdir()):
          match = re.match("([0-9]*)_([0-9]*)[.]png", filename.name)
          x = int(match.group(1))
          y = int(match.group(2))

          lst.append(DeepZoomFile(sample=self.SlideID, zoom=zoom, x=x, y=y, marker=layer, name=filename))

    lst.sort()
    writetable(self.deepzoomfolder/"zoomlist.csv", lst)

  def deepzoom(self):
    for layer in self.layers:
      self.deepzoom_vips(layer)
      self.prunezoom(layer)
      self.patchzoom(layer)
    self.writezoomlist()

@functools.total_ordering
@dataclasses.dataclass
class DeepZoomFile:
  sample: str
  zoom: int
  marker: int
  x: int
  y: int
  name: pathlib.Path = pathfield()

  def __lt__(self, other):
    return (self.sample, self.zoom, self.marker, self.x, self.y) < (other.sample, other.zoom, other.marker, other.x, other.y)
