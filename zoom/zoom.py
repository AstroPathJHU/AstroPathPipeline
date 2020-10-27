import cv2, itertools, methodtools, numpy as np, PIL, skimage

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
    return floattoint(-((-maxxy) // (self.__tilesize*onepixel)))
  def zoom(self, fmax=50):
    onepixel = units.Distance(pixels=1, pscale=self.pscale)
    #minxy = np.min([units.nominal_values(field.pxvec) for field in self.rectangles], axis=0)
    bigimage = np.zeros(shape=(len(self.layers),)+tuple(reversed(self.ntiles * self.__tilesize)), dtype=np.uint8)
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
    for tilen, (tilex, tiley) in enumerate(itertools.product(range(self.ntiles[0]), range(self.ntiles[1]))):
      xmin = tilex * self.__tilesize
      xmax = (tilex+1) * self.__tilesize
      ymin = tiley * self.__tilesize
      ymax = (tiley+1) * self.__tilesize
      slc = bigimage[:, ymin:ymax, xmin:xmax]
      if not np.any(slc): continue
      for layer in self.layers:
        filename = self.zoomfolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-X{tilex}-Y{tiley}-big.png"
        self.logger.info(f"saving {filename.name}")
        image = PIL.Image.fromarray(slc[layer-1])
        image.save(filename, "PNG")

    self.wsifolder.mkdir(parents=True, exist_ok=True)
    for layer in self.layers:
      filename = self.wsifolder/f"{self.SlideID}-Z{self.zmax}-L{layer}-wsi.png"
      self.logger.info(f"saving {filename.name}")
      image = PIL.Image.fromarray(bigimage[layer-1])
      image.save(filename, "PNG")

    return bigimage
