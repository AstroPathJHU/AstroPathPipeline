from ..baseclasses.sample import DbloadSampleBase

class FieldWithTiffMultiLayer(Field, RectangleWithTiffMultiLayer):
  pass

class AssembleImage(ReadRectanglesBase):
  def __init__(self, *args, imagesize=16384, **kwargs):
    self.__imagesize = imagesize
    super().__init__(*args, **kwargs)
  @property
  def imagesize(self): return self.__imagesize
  @property
  def rectanglecsv(self): return "fields"
  @property
  def rectangletype(self): return FieldWithImageMultiLayer
  def assembleimage(self):
    onepixel = units.Distance(pixels=1, pscale=self.pscale)
    minxy = np.min(units.pixels([field.pxvec for field in self.rectangles], axis=0), pscale=self.pscale)
    maxxy = np.max(units.pixels([field.pxvec+field.shape for field in self.rectangles], axis=0), pscale=self.pscale)
    npatches = -((-maxxy) // (self.__imagesize*onepixel)
    bigimage = np.zeros(shape=npatches * self.__imagesize, dtype=np.int8)
    for field in self.rectangles:
      with field.using_image() as image:
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
          )
          flags=cv2.INTER_CUBIC,
          borderMode=cv2.BORDER_REPLICATE,
          dsize=shifted.T.shape,
        )
        newlocalx1 = localx1 + shiftby[0]
        newlocaly1 = localy1 + shiftby[1]
        newlocalx2 = localx2 + shiftby[0]
        newlocaly2 = localy2 + shiftby[1]
        bigimage[
          globaly1/onepixel:globaly2/onepixel,
          globalx1/onepixel:globalx2/onepixel,
        ] = image[
          floattoint(localy1/onepixel):floattoint(localy2/onepixel),
          floattoint(localx1/onepixel):floattoint(localx2/onepixel),
        ]          
