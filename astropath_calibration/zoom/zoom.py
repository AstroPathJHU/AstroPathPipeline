import argparse, contextlib, cv2, itertools, methodtools, numpy as np, os, pathlib, PIL, skimage

from ..alignment.field import Field, FieldReadComponentTiffMultiLayer
from ..baseclasses.sample import ReadRectanglesBase, ReadRectanglesComponentTiff, TempDirSample, ZoomFolderSampleBase
from ..utilities import units
from ..utilities.misc import floattoint, memmapcontext, PILmaximagepixels

class ZoomSampleBase(ReadRectanglesBase):
  rectanglecsv = "fields"
  rectangletype = Field
  def __init__(self, *args, zoomtilesize=16384, **kwargs):
    self.__tilesize = zoomtilesize
    super().__init__(*args, **kwargs)
  @property
  def zoomtilesize(self): return self.__tilesize
  @methodtools.lru_cache()
  @property
  def ntiles(self):
    maxxy = np.max([units.nominal_values(field.pxvec)+field.shape for field in self.rectangles], axis=0)
    return floattoint(-((-maxxy) // (self.zoomtilesize*self.onepixel)))
  def PILmaximagepixels(self):
    return PILmaximagepixels(int(np.product(self.ntiles)) * self.__tilesize**2)

class ZoomSample(ZoomSampleBase, ZoomFolderSampleBase, TempDirSample):
  pass

class Zoom(ZoomSample, ReadRectanglesComponentTiff):
  rectangletype = FieldReadComponentTiffMultiLayer

  @property
  def logmodule(self): return "zoom"

  def zoom_wsi_fast(self, fmax=50, usememmap=False):
    onepixel = self.onepixel
    self.logger.info("allocating memory for the global array")
    with contextlib.ExitStack() as stack:
      if usememmap:
        bigimage = np.ndarray(shape=len(self.layers), dtype=object)
        for i, layer in enumerate(self.layers):
          tempfile = self.enter_context(self.tempfile())
          bigimage[i] = stack.enter_context(
            memmapcontext(
              tempfile,
              shape=tuple((self.ntiles * self.zoomtilesize)[::-1]),
              dtype=np.uint8,
              mode="w+",
            )
          )
          bigimage[i][:] = 0
      else:
        bigimage = np.zeros(shape=(len(self.layers),)+tuple((self.ntiles * self.zoomtilesize)[::-1]), dtype=np.uint8)

      nrectangles = len(self.rectangles)
      self.logger.info("assembling the global array")
      for i, field in enumerate(self.rectangles, start=1):
        self.logger.debug("%d / %d", i, nrectangles)
        with field.using_image() as image:
          image = skimage.img_as_ubyte(np.clip(image/fmax, a_min=None, a_max=1))
          globalx1 = field.mx1 // onepixel * onepixel
          globalx2 = field.mx2 // onepixel * onepixel
          globaly1 = field.my1 // onepixel * onepixel
          globaly2 = field.my2 // onepixel * onepixel
          localx1 = field.mx1 - field.px
          localx2 = localx1 + globalx2 - globalx1
          localy1 = field.my1 - field.py
          localy2 = localy1 + globaly2 - globaly1

          shiftby = np.array([globalx1 - localx1, globaly1 - localy1]) % onepixel

          newlocalx1 = localx1 + shiftby[0]
          newlocaly1 = localy1 + shiftby[1]
          newlocalx2 = localx2 + shiftby[0]
          newlocaly2 = localy2 + shiftby[1]

          for i, layer in enumerate(image):
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

            bigimage[i][
              floattoint(globaly1/onepixel):floattoint(globaly2/onepixel),
              floattoint(globalx1/onepixel):floattoint(globalx2/onepixel),
            ] = shifted[
              floattoint(newlocaly1/onepixel):floattoint(newlocaly2/onepixel),
              floattoint(newlocalx1/onepixel):floattoint(newlocalx2/onepixel),
            ]

      self.zoomfolder.mkdir(parents=True, exist_ok=True)
      ntiles = np.product(self.ntiles)
      for tilen, (tilex, tiley) in enumerate(itertools.product(range(self.ntiles[0]), range(self.ntiles[1])), start=1):
        xmin = tilex * self.zoomtilesize
        xmax = (tilex+1) * self.zoomtilesize
        ymin = tiley * self.zoomtilesize
        ymax = (tiley+1) * self.zoomtilesize
        slc = [layer[ymin:ymax, xmin:xmax] for layer in bigimage]
        if not np.any(slc):
          self.logger.info(f"       tile {tilen} / {ntiles} is empty")
          continue
        for i, layer in enumerate(self.layers):
          filename = self.zoomfilename(layer, tilex, tiley)
          self.logger.info(f"saving tile {tilen} / {ntiles} {filename.name}")
          image = PIL.Image.fromarray(slc[i])
          image.save(filename, "PNG")

      self.wsifolder.mkdir(parents=True, exist_ok=True)
      for i, layer in enumerate(self.layers):
        filename = self.wsifilename(layer)
        self.logger.info(f"saving {filename.name}")
        image = PIL.Image.fromarray(bigimage[i])
        image.save(filename, "PNG")

  def zoom_memory(self, fmax=50):
    onepixel = self.onepixel
    #minxy = np.min([units.nominal_values(field.pxvec) for field in self.rectangles], axis=0)
    buffer = -(-self.rectangles[0].shape // onepixel).astype(int) * onepixel
    nrectangles = len(self.rectangles)
    ntiles = np.product(self.ntiles)
    self.zoomfolder.mkdir(parents=True, exist_ok=True)

    class Tile(contextlib.ExitStack):
      def __init__(self, tilex, tiley, tilesize, bufferx, buffery):
        super().__init__()
        self.tilex = tilex
        self.tiley = tiley
        self.tilesize = tilesize
        self.bufferx = bufferx
        self.buffery = buffery

      @property
      def xmin(self): return self.tilex * self.tilesize * onepixel - self.bufferx
      @property
      def xmax(self): return (self.tilex+1) * self.tilesize * onepixel + self.bufferx
      @property
      def ymin(self): return self.tiley * self.tilesize * onepixel - self.buffery
      @property
      def ymax(self): return (self.tiley+1) * self.tilesize * onepixel + self.buffery
      @property
      def primaryxmin(self): return self.tilex * self.tilesize * onepixel
      @property
      def primaryxmax(self): return (self.tilex+1) * self.tilesize * onepixel
      @property
      def primaryymin(self): return self.tiley * self.tilesize * onepixel
      @property
      def primaryymax(self): return (self.tiley+1) * self.tilesize * onepixel

      def overlapsrectangle(self, globalx1, globalx2, globaly1, globaly2):
        return globalx1 < self.primaryxmax and globalx2 > self.primaryxmin and globaly1 < self.primaryymax and globaly2 > self.primaryymin

    with contextlib.ExitStack() as stack:
      tiles = [
        stack.enter_context(
          Tile(
            tilex=tilex, tiley=tiley, tilesize=self.zoomtilesize, bufferx=buffer[0], buffery=buffer[1]
          )
        )
        for tilex, tiley in itertools.product(range(self.ntiles[0]), range(self.ntiles[1]))
      ]

      for tilen, tile in enumerate(tiles, start=1):
        with tile:
          tileimage = None

          self.logger.info("assembling tile %d / %d", tilen, ntiles)
          xmin = tile.tilex * self.zoomtilesize * onepixel - buffer[0]
          #xmax = (tilex+1) * self.zoomtilesize * onepixel + buffer[0]
          ymin = tile.tiley * self.zoomtilesize * onepixel - buffer[1]
          #ymax = (tiley+1) * self.zoomtilesize * onepixel + buffer[1]

          for i, field in enumerate(self.rectangles, start=1):
            self.logger.debug("  rectangle %d / %d", i, nrectangles)

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

            tilex1 = globalx1 - xmin
            tilex2 = globalx2 - xmin
            tiley1 = globaly1 - ymin
            tiley2 = globaly2 - ymin
            localx1 = field.mx1 - field.px
            localx2 = localx1 + tilex2 - tilex1
            localy1 = field.my1 - field.py
            localy2 = localy1 + tiley2 - tiley1

            if tileimage is None: tileimage = np.zeros(shape=(len(self.layers),)+tuple((self.zoomtilesize + 2*floattoint(buffer/onepixel))[::-1]), dtype=np.uint8)

            with field.using_image() as image:
              image = skimage.img_as_ubyte(np.clip(image/fmax, a_min=None, a_max=1))

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
                ) for layer in image
              ])
              newlocalx1 = localx1 + shiftby[0]
              newlocaly1 = localy1 + shiftby[1]
              newlocalx2 = localx2 + shiftby[0]
              newlocaly2 = localy2 + shiftby[1]

              kw = {"atol": 1e-7}
              tileimage[
                :,
                floattoint(tiley1/onepixel, **kw):floattoint(tiley2/onepixel, **kw),
                floattoint(tilex1/onepixel, **kw):floattoint(tilex2/onepixel, **kw),
              ] = shifted[
                :,
                floattoint(newlocaly1/onepixel, **kw):floattoint(newlocaly2/onepixel, **kw),
                floattoint(newlocalx1/onepixel, **kw):floattoint(newlocalx2/onepixel, **kw),
              ]

        if tileimage is None: continue
        slc = tileimage[
          :,
          floattoint(buffer[1]/self.onepixel):floattoint(-buffer[1]/self.onepixel),
          floattoint(buffer[0]/self.onepixel):floattoint(-buffer[0]/self.onepixel),
        ]
        if not np.any(slc): continue
        for layer in self.layers:
          filename = self.zoomfilename(layer, tile.tilex, tile.tiley)
          self.logger.info(f"  saving {filename.name}")
          image = PIL.Image.fromarray(slc[layer-1])
          image.save(filename, "PNG")

  def wsi_vips(self):
    import pyvips

    self.wsifolder.mkdir(parents=True, exist_ok=True)
    for layer in self.layers:
      images = []
      blank = None
      for tiley, tilex in itertools.product(range(self.ntiles[1]), range(self.ntiles[0])):
        filename = self.zoomfilename(layer, tilex, tiley)
        if filename.exists():
          images.append(pyvips.Image.new_from_file(os.fspath(filename)))
        else:
          if blank is None:
            blank = pyvips.Image.new_from_memory(np.zeros(shape=(self.zoomtilesize*self.zoomtilesize,), dtype=np.uint8), width=self.zoomtilesize, height=self.zoomtilesize, bands=1, format="uchar")
          images.append(blank)

      filename = self.wsifilename(layer)
      self.logger.info(f"saving {filename.name}")
      output = pyvips.Image.arrayjoin(images, across=self.ntiles[0])
      output.pngsave(os.fspath(filename))

  def zoom_wsi_memory(self, fmax=50):
    self.zoom_memory(fmax=fmax)
    self.wsi_vips()

  def zoom_wsi(self, *args, mode="vips", **kwargs):
    self.logger.info(f"zoom running in {mode} mode")
    if mode == "vips":
      return self.zoom_wsi_memory(*args, **kwargs)
    elif mode == "fast":
      return self.zoom_wsi_fast(*args, **kwargs)
    elif mode == "memmap":
      return self.zoom_wsi_fast(*args, usememmap=True, **kwargs)
    else:
      raise ValueError(f"Bad mode {mode}")

def main(args=None):
  p = argparse.ArgumentParser()
  p.add_argument("root1", type=pathlib.Path)
  p.add_argument("root2", type=pathlib.Path)
  p.add_argument("zoomroot", type=pathlib.Path)
  p.add_argument("samp")
  p.add_argument("--units", choices=("fast", "safe"), default="fast")
  p.add_argument("--mode", choices=("vips", "fast", "memmap"), default="vips")
  args = p.parse_args(args=args)

  units.setup(args.units)

  return Zoom(root=args.root1, root2=args.root2, samp=args.samp, zoomroot=args.zoomroot).zoom_wsi(mode=args.mode)

if __name__ == "__main__":
  main()
