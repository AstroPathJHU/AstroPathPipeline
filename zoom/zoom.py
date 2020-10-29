import contextlib, cv2, itertools, methodtools, numpy as np, os, PIL, skimage

from ..alignment.field import Field
from ..baseclasses.rectangle import RectangleReadComponentTiffMultiLayer
from ..baseclasses.sample import ReadRectanglesComponentTiff
from ..utilities import units
from ..utilities.misc import floattoint

class FieldReadComponentTiffMultiLayer(Field, RectangleReadComponentTiffMultiLayer):
  pass

class Zoom(ReadRectanglesComponentTiff):
  rectanglecsv = "fields"
  rectangletype = FieldReadComponentTiffMultiLayer
  def __init__(self, *args, zoomroot, tilesize=16384, **kwargs):
    self.__tilesize = tilesize
    self.__zoomroot = zoomroot
    super().__init__(*args, **kwargs)
  @property
  def zoomroot(self): return self.__zoomroot
  @property
  def zoomfolder(self): return self.zoomroot/self.SlideID/"big"
  @property
  def wsifolder(self): return self.zoomroot/self.SlideID/"wsi"
  @property
  def tilesize(self): return self.__tilesize
  @property
  def zmax(self): return 9
  @property
  def logmodule(self): return "zoom"
  @methodtools.lru_cache()
  @property
  def ntiles(self):
    onepixel = units.Distance(pixels=1, pscale=self.pscale)
    maxxy = np.max([units.nominal_values(field.pxvec)+field.shape for field in self.rectangles], axis=0)
    return floattoint(-((-maxxy) // (self.tilesize*onepixel)))

  def zoom_wsi_fast(self, fmax=50):
    onepixel = units.Distance(pixels=1, pscale=self.pscale)
    #minxy = np.min([units.nominal_values(field.pxvec) for field in self.rectangles], axis=0)
    bigimage = np.zeros(shape=(len(self.layers),)+tuple((self.ntiles * self.tilesize)[::-1]), dtype=np.uint8)
    nrectangles = len(self.rectangles)
    for i, field in enumerate(self.rectangles, start=1):
      self.logger.info("%d / %d", i, nrectangles)
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

        bigimage[
          :,
          floattoint(globaly1/onepixel):floattoint(globaly2/onepixel),
          floattoint(globalx1/onepixel):floattoint(globalx2/onepixel),
        ] = shifted[
          :,
          floattoint(newlocaly1/onepixel):floattoint(newlocaly2/onepixel),
          floattoint(newlocalx1/onepixel):floattoint(newlocalx2/onepixel),
        ]

    self.zoomfolder.mkdir(parents=True, exist_ok=True)
    ntiles = np.product(self.ntiles)
    for tilen, (tilex, tiley) in enumerate(itertools.product(range(self.ntiles[0]), range(self.ntiles[1])), start=1):
      xmin = tilex * self.tilesize
      xmax = (tilex+1) * self.tilesize
      ymin = tiley * self.tilesize
      ymax = (tiley+1) * self.tilesize
      slc = bigimage[:, ymin:ymax, xmin:xmax]
      if not np.any(slc): continue
      for layer in self.layers:
        filename = self.zoomfolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-X{tilex}-Y{tiley}-big.png"
        self.logger.info(f"saving tile {tilen} / {ntiles} {filename.name}")
        image = PIL.Image.fromarray(slc[layer-1])
        image.save(filename, "PNG")

    self.wsifolder.mkdir(parents=True, exist_ok=True)
    for layer in self.layers:
      filename = self.wsifolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-wsi.png"
      self.logger.info(f"saving {filename.name}")
      image = PIL.Image.fromarray(bigimage[layer-1])
      image.save(filename, "PNG")

    return bigimage

  def zoom_memory(self, fmax=50):
    onepixel = units.Distance(pixels=1, pscale=self.pscale)
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
            tilex=tilex, tiley=tiley, tilesize=self.tilesize, bufferx=buffer[0], buffery=buffer[1]
          )
        )
        for tilex, tiley in itertools.product(range(self.ntiles[0]), range(self.ntiles[1]))
      ]

      for tilen, tile in enumerate(tiles, start=1):
        with tile:
          tileimage = None

          self.logger.info("tile %d / %d", tilen, ntiles)
          xmin = tile.tilex * self.tilesize * onepixel - buffer[0]
          #xmax = (tilex+1) * self.tilesize * onepixel + buffer[0]
          ymin = tile.tiley * self.tilesize * onepixel - buffer[1]
          #ymax = (tiley+1) * self.tilesize * onepixel + buffer[1]

          for i, field in enumerate(self.rectangles, start=1):
            self.logger.info("  rectangle %d / %d", i, nrectangles)

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

            if tileimage is None: tileimage = np.zeros(shape=(len(self.layers),)+tuple((self.tilesize + 2*units.pixels(buffer, pscale=self.pscale))[::-1]), dtype=np.uint8)

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

              tileimage[
                :,
                floattoint(tiley1/onepixel):floattoint(tiley2/onepixel),
                floattoint(tilex1/onepixel):floattoint(tilex2/onepixel),
              ] = shifted[
                :,
                floattoint(newlocaly1/onepixel):floattoint(newlocaly2/onepixel),
                floattoint(newlocalx1/onepixel):floattoint(newlocalx2/onepixel),
              ]

        slc = tileimage[
          :,
          units.pixels(buffer[1], pscale=self.pscale):units.pixels(-buffer[1], pscale=self.pscale),
          units.pixels(buffer[0], pscale=self.pscale):units.pixels(-buffer[0], pscale=self.pscale),
        ]
        if not np.any(slc): continue
        for layer in self.layers:
          filename = self.zoomfolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-X{tile.tilex}-Y{tile.tiley}-big.png"
          self.logger.info(f"  saving {filename.name}")
          image = PIL.Image.fromarray(slc[layer-1])
          image.save(filename, "PNG")

  def wsi_vips(self):
    import pyvips

    self.wsifolder.mkdir(parents=True, exist_ok=True)
    for layer in self.layers:
      images = []
      blank = None
      for tilex, tiley in itertools.product(range(self.ntiles[0]), range(self.ntiles[1])):
        filename = self.wsifolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-X{tilex}-Y{tiley}-big.png"
        if filename.exists():
          images.append(pyvips.Image.new_from_file(os.fspath(filename)))
        else:
          if blank is None:
            blank = pyvips.Image.new_from_memory(np.zeros(shape=(self.tilesize*self.tilesize,), dtype=np.uint8), width=self.tilesize, height=self.tilesize, bands=1, format="uchar")
          images.append(blank)

      filename = self.wsifolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-wsi.png"
      self.logger.info(f"saving {filename.name}")
      output = pyvips.Image.arrayjoin(images, across=self.ntiles[0])
      output.pngsave(os.fspath(filename))

  def zoom_wsi_memory(self, fmax=50):
    self.zoom_memory(fmax=fmax)
    self.wsi_vips()

  def zoom_wsi(self, *args, fast=False, **kwargs):
    return (self.zoom_wsi_fast if fast else self.zoom_wsi_memory)(*args, **kwargs)
