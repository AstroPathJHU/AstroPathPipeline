import contextlib, cv2, datetime, itertools, job_lock, jxmlease, methodtools, numpy as np, os, pathlib, PIL, re, shutil, skimage.transform

from ...shared.argumentparser import CleanupArgumentParser, SelectLayersArgumentParser
from ...shared.sample import ReadRectanglesDbloadComponentTiff, TempDirSample, WorkflowSample, ZoomFolderSampleBase
from ...utilities.miscfileio import memmapcontext, rm_missing_ok
from ...utilities.miscimage import vips_image_to_array
from ...utilities.miscmath import floattoint
from ..align.alignsample import AlignSample
from ..align.field import FieldReadComponentTiffMultiLayer
from ..stitchmask.stitchmasksample import AstroPathTissueMaskSample
from .zoomsamplebase import ZoomSampleBase

class ZoomSample(AstroPathTissueMaskSample, ZoomSampleBase, ZoomFolderSampleBase, TempDirSample, ReadRectanglesDbloadComponentTiff, WorkflowSample, CleanupArgumentParser, SelectLayersArgumentParser):
  """
  Run the zoom step of the pipeline:
  create big images of 16384x16384 pixels by merging the fields
  using the primary areas (mx1, my1) to (mx2, my2) defined by the
  alignment step, and merge them into a wsi image that shows the
  whole slide.

  There are three modes:
    1. fast assembles the image in memory
    2. memmap assembles the image in a memmap in a temp directory
    3. vips assembles each 16384x16384 tile in memory and uses
       libvips to merge them together into the wsi
  """
  rectangletype = FieldReadComponentTiffMultiLayer

  def __init__(self, *args, tifflayers="color", **kwargs):
    self.__tifflayers = tifflayers
    super().__init__(*args, **kwargs)

  @property
  def tifflayers(self): return self.__tifflayers

  @classmethod
  def logmodule(self): return "zoom"

  def zoom_wsi_fast(self, *, fmax=50, tissuemeanintensity=100., usememmap=False):
    """
    Assemble the wsi images either in memory (usememmap=False)
    or in a memmap in a temp directory (usememmap=True)

    fmax: initial scaling factor when loading the image
    tissuemeanintensity (default: ): the image is rescaled so that the
    average intensity in tissue regions is this number
    """
    try:
      import pyvips
    except ImportError:
      raise ImportError("Please pip install pyvips to use this functionality")

    onepixel = self.onepixel
    self.logger.info("allocating memory for the global array")
    bigimageshape = tuple((self.ntiles * self.zoomtilesize)[::-1]) + (len(self.layers),)
    fortiff = []
    with contextlib.ExitStack() as stack:
      if usememmap:
        tempfile = stack.enter_context(self.tempfile())
        bigimage = stack.enter_context(
          memmapcontext(
            tempfile,
            shape=bigimageshape,
            dtype=np.uint8,
            mode="w+",
          )
        )
        bigimage[:] = 0
      else:
        bigimage = np.zeros(shape=bigimageshape, dtype=np.uint8)

      #loop over HPFs and fill them into the big image
      nrectangles = len(self.rectangles)
      self.logger.info("assembling the global array")
      for i, field in enumerate(self.rectangles, start=1):
        self.logger.debug("%d / %d", i, nrectangles)
        #load the image file
        with field.using_image() as image:
          #scale the intensity
          image = skimage.img_as_ubyte(np.clip(image/fmax, a_min=None, a_max=1))

          #find where it should sit in the wsi
          globalx1 = field.mx1 // onepixel * onepixel
          globalx2 = field.mx2 // onepixel * onepixel
          globaly1 = field.my1 // onepixel * onepixel
          globaly2 = field.my2 // onepixel * onepixel
          #and what part of the image we should take
          localx1 = field.mx1 - field.px
          localx2 = localx1 + globalx2 - globalx1
          localy1 = field.my1 - field.py
          localy2 = localy1 + globaly2 - globaly1

          #shift the image by the fractional pixel difference between global
          #and local coordinates
          shiftby = np.array([globalx1 - localx1, globaly1 - localy1]) % onepixel

          #new local coordinates after the shift
          #they should be integer pixels (will be checked when filling the image)
          newlocalx1 = localx1 + shiftby[0]
          newlocaly1 = localy1 + shiftby[1]
          newlocalx2 = localx2 + shiftby[0]
          newlocaly2 = localy2 + shiftby[1]

          for i, layer in enumerate(image.transpose(2, 0, 1)):
            shifted = cv2.warpAffine(
              layer,
              np.array(
                [
                  [1, 0, shiftby[0]/onepixel],
                  [0, 1, shiftby[1]/onepixel],
                ],
                dtype=float,
              ),
              flags=cv2.INTER_CUBIC,
              borderMode=cv2.BORDER_REPLICATE,
              dsize=layer.T.shape,
            )

            if globaly1 < 0:
              newlocaly1 -= globaly1
              globaly1 -= globaly1
            if globalx1 < 0:
              newlocalx1 -= globalx1
              globalx1 -= globalx1
            #fill the big image with the HPF image
            kw = {"atol": 1e-7}
            bigimage[
              floattoint(float(globaly1/onepixel), **kw):floattoint(float(globaly2/onepixel), **kw),
              floattoint(float(globalx1/onepixel), **kw):floattoint(float(globalx2/onepixel), **kw),
              i,
            ] = shifted[
              floattoint(float(newlocaly1/onepixel), **kw):floattoint(float(newlocaly2/onepixel), **kw),
              floattoint(float(newlocalx1/onepixel), **kw):floattoint(float(newlocalx2/onepixel), **kw),
            ]

      #rescale the image intensity
      with self.using_tissuemask() as mask:
        meanintensity = np.mean(bigimage[:,:,0][mask])
        bigimage = (bigimage * (tissuemeanintensity / meanintensity)).clip(0, 255).astype(np.uint8)

      #save the wsi
      self.wsifolder.mkdir(parents=True, exist_ok=True)
      for i, layer in enumerate(self.layers):
        filename = self.wsifilename(layer)
        self.logger.info(f"saving {filename.name}")
        slc = bigimage[:, :, i]
        image = PIL.Image.fromarray(slc)
        image.save(filename, "PNG")

        if layer in self.needtifflayers:
          contiguousslc = np.ascontiguousarray(slc)
          vipsimage = pyvips.Image.new_from_memory(contiguousslc, width=bigimage.shape[1], height=bigimage.shape[0], bands=1, format="uchar")
          fortiff.append(vipsimage)

      self.makewsitiff(fortiff)

  @methodtools.lru_cache()
  @staticmethod
  def _colormatrix(*, dtype):
    here = pathlib.Path(__file__).parent
    with open(here/"color_matrix.txt") as f:
      matrix = re.search(r"(?<=\[).*(?=\])", f.read(), re.DOTALL).group(0)
    return np.array([[float(_) for _ in row.split()] for row in matrix.split(";")], dtype=dtype)

  @property
  def colormatrix(self): return self._colormatrix(dtype=np.float16)[tuple(np.array(self.layers)-1), :]

  @property
  def needtifflayers(self):
    if self.tifflayers is None: return ()
    if self.tifflayers == "color": return range(1, 9)
    return self.tifflayers

  def makewsitiff(self, layers):
    if not self.tifflayers:
      return
    filename = self.wsitifffilename(self.tifflayers)
    if filename.exists():
      self.logger.info(f"{filename.name} already exists")
      return

    inputlayers = layers
    self.logger.info(f"saving {filename.name}")
    scale = 2**(self.ztiff-self.zmax)
    if scale == 1:
      pass
    else:
      self.logger.info(f"  rescaling by {scale}")
      layers = [layer.resize(scale, vscale=scale) for layer in layers]

    if self.tifflayers == "color":
      self.logger.info("  normalizing")
      layerarrays = np.zeros(dtype=np.float16, shape=(len(layers),)+tuple(floattoint(self.ntiles * self.zoomtilesize * scale)[::-1]))
      import pyvips
      pyvips.cache_set_max(0)
      for i, layer in enumerate(layers):
        self.logger.debug(f"    layer {i+1} / {len(layers)}")
        layerarrays[i] = vips_image_to_array(layer)
        layers[i] = inputlayers[i] = None
      layerarrays /= np.max(layerarrays, axis=(1, 2))[:, np.newaxis, np.newaxis]
      layerarrays = 180 * np.sinh(1.5 * layerarrays)
      self.logger.info("  multiplying by color matrix")
      img = np.tensordot(layerarrays, self.colormatrix, [[0], [0]])
      img = img.clip(0, 255)
      pilimage = PIL.Image.fromarray(img.astype(np.uint8))
      self.logger.info("  saving")
      pilimage.save(filename)
    else:
      assert len(layers) == len(self.tifflayers)
      tiffoutput = layers[0]
      for layer in layers[1:]:
        tiffoutput = tiffoutput.join(layer, "vertical")
      self.logger.info("  saving")
      tiffoutput.tiffsave(os.fspath(filename), page_height=layers[0].height)

  def zoom_memory(self, *, fmax=50, tissuemeanintensity=100.):
    """
    Run zoom by saving one big tile at a time
    (afterwards you can call wsi_vips to save the wsi)
    """
    onepixel = self.onepixel
    buffer = -(-self.rectangles[0].shape // onepixel).astype(int) * onepixel
    nrectangles = len(self.rectangles)
    ntiles = np.product(self.ntiles)
    self.bigfolder.mkdir(parents=True, exist_ok=True)
    meanintensitynumerator = meanintensitydenominator = 0

    class Tile(contextlib.ExitStack):
      """
      Helper class to save the big tiles.

      The class also inherits from ExitStack so that you can use it
      to enter using_image contexts for rectangles.
      """
      def __init__(self, tilex, tiley, tilesize, bufferx, buffery):
        super().__init__()
        self.tilex = tilex
        self.tiley = tiley
        self.tilesize = tilesize
        self.bufferx = bufferx
        self.buffery = buffery

      @property
      def primaryxmin(self): return self.tilex * self.tilesize * onepixel
      @property
      def primaryxmax(self): return (self.tilex+1) * self.tilesize * onepixel
      @property
      def primaryymin(self): return self.tiley * self.tilesize * onepixel
      @property
      def primaryymax(self): return (self.tiley+1) * self.tilesize * onepixel

      def overlapsrectangle(self, globalx1, globalx2, globaly1, globaly2):
        """
        Does this tile overlap the rectangle with the given global coordinates?
        """
        return globalx1 < self.primaryxmax and globalx2 > self.primaryxmin and globaly1 < self.primaryymax and globaly2 > self.primaryymin

    #how this works in terms of context managers:

    #we first have an ExitStack that __enter__s ALL the tiles
    #for each tile:
    #  with tile:
    #    for each rectangle that overlaps the tile:
    #      for each other tile that we haven't gotten to yet that overlaps the rectangle:
    #        othertile.enter_context(rectangle.using_image())
    #      with rectangle.using_image():
    #        fill the image into this tile

    #in this way, we load each image in the first tile that uses it,
    #and only take it out of memory after finishing the last tile that uses it.

    #the initial ExitStack is so that we call __exit__ on all the tiles,
    #even if there's an exception while processing one of the earlier ones.

    with self.using_tissuemask() as mask, contextlib.ExitStack() as stack:
      tiles = [
        stack.enter_context(
          Tile(
            tilex=tilex, tiley=tiley, tilesize=self.zoomtilesize, bufferx=buffer[0], buffery=buffer[1]
          )
        )
        for tilex, tiley in itertools.product(range(self.ntiles[0]), range(self.ntiles[1]))
      ]

      wsilayer1 = self.wsifilename(1)
      if wsilayer1.exists():
        with self.PILmaximagepixels(), PIL.Image.open(wsilayer1) as img:
          img = np.asarray(img)
          meanintensitynumerator += np.sum(img[mask])
          meanintensitydenominator += np.count_nonzero(mask)

      for tilen, tile in enumerate(tiles, start=1):
        with tile:
          tilemask = mask[
            floattoint(float(tile.primaryymin/self.onepixel)):floattoint(float(tile.primaryymax/self.onepixel)),
            floattoint(float(tile.primaryxmin/self.onepixel)):floattoint(float(tile.primaryxmax/self.onepixel)),
          ]

          for layer in self.layers:
            if self.wsifilename(layer).exists(): continue
            filename = self.bigfilename(layer, tile.tilex, tile.tiley)
            with job_lock.JobLock(filename.with_suffix(".lock"), corruptfiletimeout=datetime.timedelta(minutes=10), outputfiles=[filename], checkoutputfiles=False) as lock:
              assert lock
              if not filename.exists():
                break
          else:
            self.logger.info(f"  {self.bigfilename('*', tile.tilex, tile.tiley)} have already been zoomed")
            if not wsilayer1.exists():
              filename = self.bigfilename(1, tile.tilex, tile.tiley)
              with self.PILmaximagepixels(), PIL.Image.open(filename) as img:
                img = np.asarray(img)
                meanintensitynumerator += np.sum(img[tilemask])
                meanintensitydenominator += np.count_nonzero(tilemask)
            continue

          #tileimage is initialized to None so that we don't have
          #to initialize the big array unless there are actually
          #nonzero pixels in the tile
          tileimage = None

          self.logger.info("assembling tile %d / %d", tilen, ntiles)
          xmin = tile.tilex * self.zoomtilesize * onepixel - buffer[0]
          ymin = tile.tiley * self.zoomtilesize * onepixel - buffer[1]

          for i, field in enumerate(self.rectangles, start=1):
            self.logger.debug("  rectangle %d / %d", i, nrectangles)

            #find where it should sit in the wsi
            globalx1 = field.mx1 // onepixel * onepixel
            globalx2 = field.mx2 // onepixel * onepixel
            globaly1 = field.my1 // onepixel * onepixel
            globaly2 = field.my2 // onepixel * onepixel

            if not tile.overlapsrectangle(globalx1=globalx1, globalx2=globalx2, globaly1=globaly1, globaly2=globaly2): continue
            for othertilen, othertile in enumerate(tiles, start=1):
              #if any other tile uses the same field:
              #keep the field in memory until we get to that tile
              if othertilen == tilen: assert othertile is tile
              if othertilen <= tilen: continue
              if othertile.overlapsrectangle(globalx1=globalx1, globalx2=globalx2, globaly1=globaly1, globaly2=globaly2):
                othertile.enter_context(field.using_image())

            if tileimage is None: tileimage = np.zeros(shape=tuple((self.zoomtilesize + 2*floattoint((buffer/onepixel).astype(float)))[::-1]) + (len(self.layers),), dtype=np.uint8)

            with field.using_image() as image:
              image = skimage.img_as_ubyte(np.clip(image/fmax, a_min=None, a_max=1))

              #find where it should sit in the tile
              tilex1 = globalx1 - xmin
              tilex2 = globalx2 - xmin
              tiley1 = globaly1 - ymin
              tiley2 = globaly2 - ymin
              #and what part of the image we should take
              localx1 = field.mx1 - field.px
              localx2 = localx1 + tilex2 - tilex1
              localy1 = field.my1 - field.py
              localy2 = localy1 + tiley2 - tiley1

              #shift the image by the fractional pixel difference between tile
              #and local coordinates
              shiftby = np.array([tilex1 - localx1, tiley1 - localy1]) % onepixel

              shifted = np.array([
                cv2.warpAffine(
                  layer,
                  np.array(
                    [
                      [1, 0, shiftby[0]/onepixel],
                      [0, 1, shiftby[1]/onepixel],
                    ],
                    dtype=float,
                  ),
                  flags=cv2.INTER_CUBIC,
                  borderMode=cv2.BORDER_REPLICATE,
                  dsize=layer.T.shape,
                ) for layer in image.transpose(2, 0, 1)
              ]).transpose(1, 2, 0)

              #new local coordinates after the shift
              #they should be integer pixels (will be checked when filling the image)
              newlocalx1 = localx1 + shiftby[0]
              newlocaly1 = localy1 + shiftby[1]
              newlocalx2 = localx2 + shiftby[0]
              newlocaly2 = localy2 + shiftby[1]

              if tiley1 < 0:
                newlocaly1 -= tiley1
                tiley1 -= tiley1
              if tilex1 < 0:
                newlocalx1 -= tilex1
                tilex1 -= tilex1
              kw = {"atol": 1e-7}
              tileimage[
                floattoint(float(tiley1/onepixel), **kw):floattoint(float(tiley2/onepixel), **kw),
                floattoint(float(tilex1/onepixel), **kw):floattoint(float(tilex2/onepixel), **kw),
                :,
              ] = shifted[
                floattoint(float(newlocaly1/onepixel), **kw):floattoint(float(newlocaly2/onepixel), **kw),
                floattoint(float(newlocalx1/onepixel), **kw):floattoint(float(newlocalx2/onepixel), **kw),
                :,
              ]

        if tileimage is None: continue
        #remove the buffer
        slc = tileimage[
          floattoint(float(buffer[1]/self.onepixel)):floattoint(float(-buffer[1]/self.onepixel)),
          floattoint(float(buffer[0]/self.onepixel)):floattoint(float(-buffer[0]/self.onepixel)),
          :,
        ]
        if not np.any(slc): continue
        #increment the numerator and denominator of the
        #mean tissue intensity
        meanintensitynumerator += np.sum(slc[:,:,0][tilemask])
        meanintensitydenominator += np.count_nonzero(tilemask)

        #save the tile images
        for layer in self.layers:
          filename = self.bigfilename(layer, tile.tilex, tile.tiley)
          with job_lock.JobLock(filename.with_suffix(".lock"), corruptfiletimeout=datetime.timedelta(minutes=10), outputfiles=[filename], checkoutputfiles=False) as lock:
            assert lock
            if filename.exists():
              self.logger.info(f"  {filename.name} was already created")
              continue
            self.logger.info(f"  saving {filename.name}")
            image = PIL.Image.fromarray(slc[:, :, layer-1])
            image.save(filename, "PNG")

    return tissuemeanintensity / (meanintensitynumerator / meanintensitydenominator)

  def wsi_vips(self, scaleby):
    """
    Call vips to assemble the big images into the wsi
    """
    try:
      import pyvips
    except ImportError:
      raise ImportError("Please pip install pyvips to use this functionality")

    fortiff = []

    self.wsifolder.mkdir(parents=True, exist_ok=True)
    for layer in self.layers:
      wsifilename = self.wsifilename(layer)
      if wsifilename.exists():
        self.logger.info(f"{wsifilename.name} already exists")
        if layer in self.needtifflayers:
          output = pyvips.Image.new_from_file(os.fspath(wsifilename))
          fortiff.append(output)
        continue

      images = []
      removefilenames = []
      blank = None
      for tiley, tilex in itertools.product(range(self.ntiles[1]), range(self.ntiles[0])):
        bigfilename = self.bigfilename(layer, tilex, tiley)
        if bigfilename.exists():
          images.append(pyvips.Image.new_from_file(os.fspath(bigfilename)).linear(scaleby, 0))
          removefilenames.append(bigfilename)
        else:
          if blank is None:
            blank = pyvips.Image.new_from_memory(np.zeros(shape=(self.zoomtilesize*self.zoomtilesize,), dtype=np.uint8), width=self.zoomtilesize, height=self.zoomtilesize, bands=1, format="uchar")
          images.append(blank)

      self.logger.info(f"saving {wsifilename.name}")
      output = pyvips.Image.arrayjoin(images, across=self.ntiles[0])
      output.pngsave(os.fspath(wsifilename))
      if layer in self.needtifflayers:
        fortiff.append(output)

      for big in removefilenames:
        rm_missing_ok(big)

    shutil.rmtree(self.bigfolder)

    self.makewsitiff(fortiff)

  def zoom_wsi_memory(self, fmax=50, tissuemeanintensity=100.):
    scaleby = self.zoom_memory(fmax=fmax, tissuemeanintensity=tissuemeanintensity)
    self.wsi_vips(scaleby=scaleby)

  def zoom_wsi(self, *args, mode="vips", **kwargs):
    self.logger.info(f"zoom running in {mode} mode")
    if mode == "vips":
      return self.zoom_wsi_memory(*args, **kwargs)
    elif mode == "fast":
      return self.zoom_wsi_fast(*args, usememmap=False, **kwargs)
    elif mode == "memmap":
      return self.zoom_wsi_fast(*args, usememmap=True, **kwargs)
    else:
      raise ValueError(f"Bad mode {mode}")

  def run(self, *, cleanup=False, **kwargs):
    if cleanup: self.cleanup()
    self.zoom_wsi(**kwargs)

  def inputfiles(self, **kwargs):
    return super().inputfiles(**kwargs) + [
      *(r.imagefile for r in self.rectangles),
      self.csv("fields"),
    ]

  @property
  def workflowkwargs(self):
    return {"layers": self.layers, "tifflayers": self.tifflayers, **super().workflowkwargs}

  @property
  def workinprogressfiles(self):
    return [
      *(self.bigfolder.glob("*.png")),
      *(self.wsifolder.glob("*.png")),
      *(self.wsifolder.glob("*.tiff")),
    ]

  @classmethod
  def getoutputfiles(cls, SlideID, *, root, zoomroot, informdataroot, layers, tifflayers, **otherrootkwargs):
    nlayers = cls.getnlayersunmixed(informdataroot/SlideID/"inform_data"/"Component_Tiffs")
    if layers is None:
      layers = range(1, nlayers+1)
    result = [
      *(
        zoomroot/SlideID/"wsi"/f"{SlideID}-Z{cls.zmax}-L{layer}-wsi.png"
        for layer in layers
      ),
    ]
    if tifflayers:
      tiffname = f"{SlideID}-Z{cls.ztiff}"
      if tifflayers == "color":
        tiffname += "-color"
      elif frozenset(tifflayers) != frozenset(range(1, nlayers+1)):
        tiffname += "-L" + "".join(str(l) for l in sorted(tifflayers))
      tiffname += "-wsi.tiff"
      result.append(zoomroot/SlideID/"wsi"/tiffname)
    return result

  @classmethod
  def workflowdependencyclasses(cls):
    return [AlignSample] + super().workflowdependencyclasses()

def main(args=None):
  ZoomSample.runfromargumentparser(args)

if __name__ == "__main__":
  main()
