import collections, errno, functools, jxmlease, numpy as np, os, pathlib, PIL, re, shutil

from ...shared.argumentparser import CleanupArgumentParser, SelectLayersArgumentParser
from ...shared.sample import DbloadSampleBase, DeepZoomSampleBase, SelectLayersComponentTiff, WorkflowSample, ZoomFolderSampleBase
from ...utilities.dataclasses import MyDataClass
from ...utilities.miscfileio import rm_missing_ok
from ...utilities.tableio import pathfield, readtable, writetable
from ..zoom.zoomsample import ZoomSample

class DeepZoomSample(SelectLayersComponentTiff, DbloadSampleBase, ZoomFolderSampleBase, DeepZoomSampleBase, WorkflowSample, CleanupArgumentParser, SelectLayersArgumentParser):
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

  multilayer = True

  @classmethod
  def logmodule(self): return "deepzoom"

  @property
  def tilesize(self): return self.__tilesize

  def layerfolder(self, layer):
    """
    Folder where the image pyramid for a given layer will go
    """
    return self.deepzoomfolder/f"L{layer:d}_files"

  def deepzoom_vips(self, layer):
    """
    Use vips to create the image pyramid.  This is an out of the box
    functionality of vips.
    """
    try:
      import pyvips
    except ImportError:
      raise ImportError("Please pip install pyvips to use this functionality")
    self.logger.info("running vips for layer %d", layer)
    filename = self.wsifilename(layer)

    #create the output folder and make sure it's empty
    self.deepzoomfolder.mkdir(parents=True, exist_ok=True)
    destfolder = self.layerfolder(layer)
    if destfolder.exists():
      for subfolder in destfolder.iterdir():
        if subfolder.is_dir(): subfolder.rmdir()
      destfolder.rmdir()

    #remove "_files" from the folder name for vips, because
    #vips adds that
    dest = destfolder.with_name(destfolder.name.replace("_files", ""))

    #open the wsi in vips and save the deepzoom
    wsi = pyvips.Image.new_from_file(os.fspath(filename))
    wsi.dzsave(os.fspath(dest), suffix=".png", background=0, depth="onetile", overlap=0, tile_size=self.tilesize)

  def prunezoom(self, layer):
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
    self.logger.info("checking which files are non-empty for layer %d", layer)
    destfolder = self.layerfolder(layer)

    #save all file sizes in a dict
    filesizedict = collections.defaultdict(list)
    for nfiles, filename in enumerate(destfolder.glob("*/*.png"), start=1):
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

  def patchsmallimages(self, layer):
    """
    Also, sometimes the images on the right or bottom edges have 128 pixels
    in one of their dimensions.  We pad them to be 256x256.
    """
    self.logger.info("patching zoom image sizes for layer %d", layer)
    destfolder = self.layerfolder(layer)

    #pad images that are too small
    def tilexy(filename):
      match = re.match("([0-9]+)_([0-9]+)[.]png$", filename.name)
      return int(match.group(1)), int(match.group(2))
    def tilex(filename): return tilexy(filename)[0]
    def tiley(filename): return tilexy(filename)[1]
    for folder in sorted((_ for _ in destfolder.iterdir() if _.name != "runningflag"), key=lambda x: int(x.name)):
      #find the images that have the max x or the max y
      filenames = list(folder.glob("*.png"))
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

  def patchfolderstructure(self, layer):
    """
    Rename the folders to our desired convention: Z9 is always the most zoomed in,
    Z0 is always the most zoomed out.  If there are less than 10 different zoom
    levels produced by vips, we produce the missing levels manually by zooming
    in on the smallest one from vips.
    """
    #rename the folders
    self.logger.info("relabeling zooms for layer %d", layer)
    destfolder = self.layerfolder(layer)
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

  def writezoomlist(self):
    """
    Write the csv file that lists all the png files to load
    """
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
    """
    Run the full deepzoom pipeline
    """
    for layer in self.layers:
      folder = self.layerfolder(layer)
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

      self.deepzoom_vips(layer)
      (folder/"runningflag").touch()
      self.prunezoom(layer)
      self.patchsmallimages(layer)
      self.patchfolderstructure(layer)
      (folder/"runningflag").unlink()

    self.writezoomlist()

  def run(self, *, cleanup=False, **kwargs):
    if cleanup: self.cleanup()
    self.deepzoom(**kwargs)

  def inputfiles(self, **kwargs):
    return super().inputfiles(**kwargs) + [
      *(self.wsifilename(layer) for layer in self.layers),
    ]

  @property
  def workinprogressfiles(self):
    return [
      *self.deepzoomfolder.glob("L*_files/Z*/*.png"),
      *self.deepzoomfolder.glob("L*.dzi"),
    ]

  @property
  def workflowkwargs(self):
    return {"layers": self.layers, "tifflayers": None, **super().workflowkwargs}

  @classmethod
  def getoutputfiles(cls, SlideID, *, root, informdataroot, deepzoomroot, layers, checkimages=False, **otherworkflowkwargs):
    zoomlist = deepzoomroot/SlideID/"zoomlist.csv"
    if layers is None:
      with open(informdataroot/SlideID/"inform_data"/"Component_Tiffs"/"batch_procedure.ifp", "rb") as f:
        for path, _, node in jxmlease.parse(f, generator="AllComponents"):
          layers = range(1, int(node.xml_attrs["dim"])+1)
    result = [
      zoomlist,
      *(deepzoomroot/SlideID/f"L{layer}.dzi" for layer in layers),
    ]
    if checkimages and zoomlist.exists():
      files = readtable(zoomlist, DeepZoomFile)
      result += [file.name for file in files]
    return result

  @classmethod
  def workflowdependencyclasses(cls):
    return [ZoomSample] + super().workflowdependencyclasses()

@functools.total_ordering
class DeepZoomFile(MyDataClass):
  """
  Metadata for a png file for the zoomlist.csv

  sample: the SlideID
  zoom: the zoom level
  marker: the layer number
  x, y: the x and y index of the tile
  name: the png filename
  """
  sample: str
  zoom: int
  marker: int
  x: int
  y: int
  name: pathlib.Path = pathfield()

  def __lt__(self, other):
    """
    The ordering goes by zoom level, then layer, then x, then y.
    This is used to sort for the csv file.
    """
    return (self.sample, self.zoom, self.marker, self.x, self.y) < (other.sample, other.zoom, other.marker, other.x, other.y)

def main(args=None):
  DeepZoomSample.runfromargumentparser(args)

if __name__ == "__main__":
  main()
