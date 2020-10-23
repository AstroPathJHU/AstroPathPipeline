import cv2, itertools, numpy as np, PIL, skimage

from ..alignment.field import Field
from ..baseclasses.rectangle import RectangleReadComponentTiffMultiLayer
from ..baseclasses.sample import ReadRectangles
from ..utilities import units
from ..utilities.misc import floattoint

class FieldReadComponentTiffMultiLayer(Field, RectangleReadComponentTiffMultiLayer):
  pass

class AssembleImage(ReadRectangles):
  rectanglecsv = "fields"
  rectangletype = FieldReadComponentTiffMultiLayer
  def __init__(self, *args, zoomroot, tilesize=16384, **kwargs):
    self.__tilesize = tilesize
    self.__zoomroot = zoomroot
    super().__init__(*args, **kwargs)
  @property
  def zoomroot(self): return self.__zoomroot
  @property
  def tilesize(self): return self.__tilesize
  @property
  def zmax(self): return 9
  def assembleimage(self):
    onepixel = units.Distance(pixels=1, pscale=self.pscale)
    #minxy = np.min(units.pixels([field.pxvec for field in self.rectangles], axis=0), pscale=self.pscale)
    maxxy = np.max(units.pixels([field.pxvec+field.shape for field in self.rectangles], axis=0), pscale=self.pscale)
    ntiles = -((-maxxy) // (self.__tilesize*onepixel))
    bigimage = np.zeros(shape=(len(self.layers),)+tuple(ntiles * self.__tilesize), dtype=np.uint8)
    for field in self.rectangles:
      with field.using_image() as image:
        image = skimage.img_as_ubyte(image)

        globalx1 = field.mx1 // onepixel * onepixel
        globalx2 = field.mx2 // onepixel * onepixel
        globaly1 = field.my1 // onepixel * onepixel
        globaly2 = field.my2 // onepixel * onepixel
        localx1 = field.mx1 - field.px
        localx2 = field.mx2 - field.px
        localy1 = field.my1 - field.py
        localy2 = field.my2 - field.py

        shiftby = np.array([globalx1 - localx1, globaly1 - localy1]) % onepixel

        shifted = cv2.warpAffine(
          image,
          np.array(
            [
              [1, 0, shiftby[0]],
              [0, 1, shiftby[1]],
            ],
          ),
          flags=cv2.INTER_CUBIC,
          borderMode=cv2.BORDER_REPLICATE,
          dsize=image.T.shape,
        )
        newlocalx1 = localx1 + shiftby[0]
        newlocaly1 = localy1 + shiftby[1]
        newlocalx2 = localx2 + shiftby[0]
        newlocaly2 = localy2 + shiftby[1]
        bigimage[
          :,
          globaly1/onepixel:globaly2/onepixel,
          globalx1/onepixel:globalx2/onepixel,
        ] = shifted[
          :,
          floattoint(newlocaly1/onepixel):floattoint(newlocaly2/onepixel),
          floattoint(newlocalx1/onepixel):floattoint(newlocalx2/onepixel),
        ]

    for tilen, (tilex, tiley) in enumerate(itertools.product(range(ntiles[0]), range(ntiles[1]))):
      for layer in self.layers:
        xmin = tilex * self.__tilesize
        xmax = (tilex+1) * self.__tilesize
        ymin = tiley * self.__tilesize
        ymax = (tiley+1) * self.__tilesize
        image = PIL.Image.fromarray(bigimage[layer, ymin:ymax, xmin:xmax])
        filename = self.zoomroot/self.SlideID/f"{self.SlideID}-Z{self.zmax}-L{layer}-X{tilex}-Y{tiley}-big.png"
        image.save(filename, "PNG")
