import collections, errno, functools, itertools, numpy as np, os, PIL, re, shutil

from ...shared.argumentparser import CleanupArgumentParser, SelectLayersArgumentParser
from ...shared.sample import DbloadSampleBase, DeepZoomFolderSampleBaseTMAPerCore, SelectLayersComponentTiff, TissueSampleBase, WorkflowSample, ZoomFolderSampleBase, ZoomFolderSampleComponentTiff
from ...utilities.miscfileio import rm_missing_ok
from ...utilities.miscimage import array_to_vips_image
from ...utilities.tableio import readtable, writetable
from .deepzoomsample import DeepZoomFile

class DeepZoomSampleBaseTMAPerCore(DbloadSampleBase, ZoomFolderSampleBase, DeepZoomFolderSampleBaseTMAPerCore, WorkflowSample, TissueSampleBase, CleanupArgumentParser, SelectLayersArgumentParser):
  """
  The deepzoom step takes the whole slide image and produces an image pyramid
  of different zoom levels.
  """

  def __init__(self, *args, tilesize=256, **kwargs):
    """
    tilesize: size of the tiles at each level of the image pyramid
    """
    super().__init__(*args, **kwargs)
    self.__tilesize = tilesize

  @property
  def tilesize(self): return self.__tilesize

  def layerfolder(self, TMAcore, layer):
    """
    Folder where the image pyramid for a given layer will go
    """
    return self.deepzoomfolderTMAcore(TMAcore)/f"L{layer:d}_files"

  def deepzoom_vips(self, TMAcore, layer):
    """
    Use vips to create the image pyramid.  This is an out of the box
    functionality of vips.
    """
    row = TMAcore.core_row
    col = TMAcore.core_col
    self.logger.info("running vips for row %d column %d layer %d", row, col, layer)
    filename = self.percoreimagefile(TMAcore, layer)

    #create the output folder and make sure it's empty
    destfolder = self.layerfolder(TMAcore, layer)
    destfolder.parent.mkdir(parents=True, exist_ok=True)
    if destfolder.exists():
      for subfolder in destfolder.iterdir():
        if subfolder.is_dir(): subfolder.rmdir()
      destfolder.rmdir()

    #remove "_files" from the folder name for vips, because
    #vips adds that
    dest = destfolder.with_name(destfolder.name.replace("_files", ""))

    #open the image in vips and save the deepzoom
    array = np.load(filename)["arr_0"]
    if not np.all(array.shape == TMAcore.shape):
      raise ValueError(f"shape mismatch: shape in npz is {array.shape}, shape from core_locations.csv is {tuple(TMAcore.shape)}")
    img = array_to_vips_image(array)
    wsi = img.affine([1, 0, 0, 1], idx=TMAcore.x1, idy=TMAcore.y1)
    wsi.dzsave(os.fspath(dest), suffix=".png", background=0, depth="onetile", overlap=0, tile_size=self.tilesize)

  def prunezoom(self, TMAcore, layer):
    """
    Clean up the deepzoom output
    """

    #delete any empty images
    #in principle we could do this by opening them all in PIL and doing np.any,
    #but that would be very slow.  Instead we check the file size.  The empty
    #images will be smaller than ones that have content in them.

    #However, sometimes some images on the edge have one or both dimensions
    #as 128 instead of 256, so that also reduces the size.  We can't just delete
    #all the smallest ones.
    row = TMAcore.core_row
    col = TMAcore.core_col
    self.logger.info("checking which files are non-empty for row %d column %d layer %d", row, col, layer)
    destfolder = self.layerfolder(TMAcore, layer)

    #save all file sizes in a dict
    filesizedict = collections.defaultdict(list)
    nfiles = 0
    folders = [_ for _ in destfolder.glob("*/") if _.is_dir()]
    for folder in folders:
      firstfolder = int(folder.name) == min(int(_.name) for _ in folders)
      for nfiles, filename in enumerate(folder.glob("*.png"), start=nfiles+1):
        if firstfolder: continue #do not delete most zoomed out files
        size = filename.stat().st_size
        filesizedict[size].append(filename)

    nbad = 0

    #loop over the file sizes from small to large
    for size, files in sorted(filesizedict.items()):
      #open a random image with this file size
      with PIL.Image.open(files[0]) as im:
        if np.any(im):
          #if the image is non-empty
          if im.size == (self.tilesize, self.tilesize):
            #if the image size is 256x256 and it has content,
            #there are not going to be any empty images bigger than
            #this one, so we can break
            break
          else:
            #if the image size is not 256x256, then we need to move to
            #the next file size, because empty 256x256 might be bigger
            #than non-empty 128x128
            continue

      #delete the images with this file size
      self.logger.info("removing %d empty files with file size %d", len(files), size)
      for nbad, filename in enumerate(files, start=nbad+1):
        try:
          rm_missing_ok(filename)
        except OSError:
          rm_missing_ok(filename) #retry in case of network errors

    ngood = nfiles - nbad
    self.logger.info("there are %d remaining non-empty files", ngood)

  def patchsmallimages(self, TMAcore, layer):
    """
    Also, sometimes the images on the right or bottom edges have 128 pixels
    in one of their dimensions.  We pad them to be 256x256.
    """
    row = TMAcore.core_row
    col = TMAcore.core_col
    self.logger.info("patching zoom image sizes for row %d column %d layer %d", row, col, layer)
    destfolder = self.layerfolder(TMAcore, layer)

    #pad images that are too small
    def tilexy(filename):
      match = re.match("([0-9]+)_([0-9]+)[.]png$", filename.name)
      return int(match.group(1)), int(match.group(2))
    def tilex(filename): return tilexy(filename)[0]
    def tiley(filename): return tilexy(filename)[1]
    for folder in sorted((_ for _ in destfolder.iterdir() if _.name != "runningflag"), key=lambda x: int(x.name)):
      #find the images that have the max x or the max y
      filenames = list(folder.glob("*.png"))
      if not filenames:
        continue
      maxx = tilex(max(filenames, key=tilex))
      maxxfilenames = [_ for _ in filenames if tilex(_) == maxx]
      maxy = tiley(max(filenames, key=tiley))
      maxyfilenames = [_ for _ in filenames if tiley(_) == maxy]

      #check the edge tiles in x and y separately
      for edgefilenames in maxxfilenames, maxyfilenames:
        #pick a random file on this edge
        #the right edge all has the same width and the bottom edge
        #all has the same height, so we just need to check if one
        #non-corner file is too small in one of its dimensions
        edgefilename = edgefilenames[0]
        #if we got the corner, pick a different one
        if (
          edgefilename in maxxfilenames and edgefilename in maxyfilenames
          and len(edgefilenames) > 1
        ):
          edgefilename = edgefilenames[1]

        #open the image and check its size
        with PIL.Image.open(edgefilename) as im:
          n, m = im.size
          #if it's already big enough, don't need to do anything with this edge
          if m == self.tilesize and n == self.tilesize: continue
          if m > self.tilesize or n > self.tilesize:
            raise ValueError(f"{edgefilename} is too big {m}x{n}, expected <= {self.tilesize}x{self.tilesize}")

        #now that we know we need to resize, loop over the tiles on this edge
        for edgefilename in edgefilenames:
          with PIL.Image.open(edgefilename) as im:
            n, m = im.size
            #if it's already 256x256
            #this can only happen on the corner, if it was 128x128 and was
            #fixed by maxxfilenames and now we're encountering it in
            #maxyfilenames
            if m == self.tilesize and n == self.tilesize: continue
            if m > self.tilesize or n > self.tilesize:
              raise ValueError(f"{edgefilename} is too big {m}x{n}, expected <= {self.tilesize}x{self.tilesize}")
            im.load()
          #pad it with 0s and save it
          im = PIL.Image.fromarray(np.pad(np.asarray(im), ((0, self.tilesize-m), (0, self.tilesize-n))))
          im.save(edgefilename)

  def patchfolderstructure(self, TMAcore, layer):
    """
    Rename the folders to our desired convention: Z9 is always the most zoomed in,
    Z0 is always the most zoomed out.  If there are less than 10 different zoom
    levels produced by vips, we produce the missing levels manually by zooming
    in on the smallest one from vips.
    """
    #rename the folders
    row = TMAcore.core_row
    col = TMAcore.core_col
    self.logger.info("relabeling zooms for row %d column %d layer %d", row, col, layer)
    destfolder = self.layerfolder(TMAcore, layer)
    folders = sorted((_ for _ in destfolder.iterdir() if _.name != "runningflag"), key=lambda x: int(x.name))
    maxfolder = int(folders[-1].name)
    if maxfolder > 9:
      raise ValueError(f"Need more zoom levels than 0-9 (max from vips is {maxfolder})")
    newfolders = []
    for folder in reversed(folders):
      newnumber = int(folder.name) + 9 - maxfolder
      newfolder = destfolder/f"Z{newnumber}"
      folder.rename(newfolder)
      newfolders.append(newfolder)

    minzoomnumber = newnumber
    minzoomfolder = newfolder

    #fill in the missing zoom levels by zooming the smallest image ourselves
    if minzoomnumber:
      #load the smallest image
      smallestimagefilename = minzoomfolder/"0_0.png"
      with PIL.Image.open(smallestimagefilename) as smallestimage:
        smallestimage.load()

      for i in range(minzoomnumber):
        #create the folder
        newfolder = destfolder/f"Z{i}"
        newfolder.mkdir(exist_ok=True)
        newfilename = newfolder/"0_0.png"
        #resize the image, and scale to get the right brightness
        im = smallestimage.resize(np.asarray(smallestimage.size) // 2**(minzoomnumber-i))
        im = np.asarray(im)
        im = (im * 1.25**(minzoomnumber-i)).astype(np.uint8)
        m, n = im.shape
        #pad and save
        im = np.pad(im, ((0, self.tilesize-m), (0, self.tilesize-n)))
        im = PIL.Image.fromarray(im)
        im.save(newfilename)

  def writezoomlist(self, TMAcore):
    """
    Write the csv file that lists all the png files to load
    """
    lst = []
    row = TMAcore.core_row
    col = TMAcore.core_col
    for layer in self.layerszoom:
      folder = self.layerfolder(TMAcore, layer)
      for zoomfolder in sorted(folder.iterdir()):
        zoom = int(re.match("Z([0-9]*)", zoomfolder.name).group(1))
        for filename in sorted(zoomfolder.iterdir()):
          match = re.match("([0-9]*)_([0-9]*)[.]png", filename.name)
          x = int(match.group(1))
          y = int(match.group(2))

          lst.append(DeepZoomFileTMACore(sample=self.SlideID, zoom=zoom, x=x, y=y, marker=layer, row=row, col=col, name=filename.relative_to(self.deepzoomroot)))

    lst.sort()
    writetable(self.deepzoomfolder/"zoomlist.csv", lst)

  def deepzoom(self):
    """
    Run the full deepzoom pipeline
    """
    for TMAcore in self.TMAcores:
      for layer in self.layerszoom:
        folder = self.layerfolder(layer=layer, TMAcore=TMAcore)
        if folder.exists():
          for i in range(10):
            if (folder/f"{i}").exists():
              shutil.rmtree(folder/f"{i}")
            if (folder/f"Z{i}").exists():
              try:
                (folder/f"Z{i}").rmdir()
              except OSError as e:
                if e.errno == errno.ENOTEMPTY:
                  pass
                else:
                  raise
          if (folder/"runningflag").exists():
            for i in range(10):
              if (folder/f"Z{i}").exists():
                shutil.rmtree(folder/f"Z{i}")
            (folder/"runningflag").unlink()
          elif all((folder/f"Z{i}").exists() for i in range(10)):
            self.logger.info(f"layer {layer} has already been deepzoomed")
            continue

        self.deepzoom_vips(layer=layer, TMAcore=TMAcore)
        (folder/"runningflag").touch()
        self.prunezoom(layer=layer, TMAcore=TMAcore)
        self.patchsmallimages(layer=layer, TMAcore=TMAcore)
        self.patchfolderstructure(layer=layer, TMAcore=TMAcore)
        (folder/"runningflag").unlink()

      self.writezoomlist(TMAcore=TMAcore)

  def run(self, *, cleanup=False, **kwargs):
    if cleanup: self.cleanup()
    self.deepzoom(**kwargs)

  def inputfiles(self, **kwargs):
    return super().inputfiles(**kwargs) + [
      *(self.percoreimagefile(TMAcore) for TMAcore in self.TMAcores),
    ]

  @classmethod
  def getworkinprogressfiles(cls, SlideID, *, deepzoomroot, **workflowkwargs):
    deepzoomfolder = deepzoomroot/SlideID
    return itertools.chain(
      deepzoomfolder.glob("Core[[]*[]]/L*_files/Z*/*.png"),
      deepzoomfolder.glob("Core[[]*[]]/L*.dzi"),
      deepzoomfolder.glob("Core[[]*[]]/zoomlist.csv"),
    )

  @property
  def workflowkwargs(self):
    return {"tifflayers": None, **super().workflowkwargs}

  @classmethod
  def getoutputfiles(cls, SlideID, *, deepzoomroot, checkimages=False, **otherworkflowkwargs):
    result = []
    for core in cls.getTMAcores(SlideID=SlideID, **otherworkflowkwargs):
      folder = deepzoomroot/SlideID/f"Core[1,{core.core_row},{core.core_col}]"
      zoomlist = folder/"zoomlist.csv"
      layers = cls.getlayerszoom(SlideID=SlideID, **otherworkflowkwargs)
      result += [
        zoomlist,
        *(folder/f"L{layer}.dzi" for layer in layers),
      ]
      if checkimages and zoomlist.exists():
        files = readtable(zoomlist, DeepZoomFile)
        result += [deepzoomroot/file.name for file in files]
    return result

@functools.total_ordering
class DeepZoomFileTMACore(DeepZoomFile):
  row: int
  col: int

  def __lt__(self, other):
    """
    The ordering goes by zoom level, then layer, then x, then y.
    This is used to sort for the csv file.
    """
    return (self.sample, self.row, self.col, self.zoom, self.marker, self.x, self.y) < (other.sample, self.row, self.col, other.zoom, other.marker, other.x, other.y)

class DeepZoomSampleTMAPerCore(DeepZoomSampleBaseTMAPerCore, ZoomFolderSampleComponentTiff, SelectLayersComponentTiff):
  def __init__(self, *args, layers=None, **kwargs):
    super().__init__(*args, layerscomponenttiff=layers, **kwargs)

  multilayercomponenttiff = True

  @classmethod
  def logmodule(self): return "deepzoompercore"

  @property
  def workflowkwargs(self):
    return {"layers": self.layerscomponenttiff, **super().workflowkwargs}

  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return super().workflowdependencyclasses(**kwargs)

def main(args=None):
  DeepZoomSampleTMAPerCore.runfromargumentparser(args)

if __name__ == "__main__":
  main()
