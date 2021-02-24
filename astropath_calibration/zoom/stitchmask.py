import abc, contextlib, numpy as np, pathlib
from ..alignment.field import FieldReadComponentTiff
from ..baseclasses.sample import MaskSampleBase, ReadRectanglesDbloadComponentTiff
from ..utilities.misc import floattoint
from ..zoom.zoom import ZoomSampleBase

class MaskSample(MaskSampleBase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__using_mask_count = 0

  @abc.abstractproperty
  def maskfilestem(self): pass

  def maskfilename(self, folder=None, filename=None, filetype=".npz"):
    if filename is filetype is None:
      raise ValueError("Have to give either a filename or a filetype")

    if filename is None:
      stem = pathlib.Path(self.maskfilestem)
      if stem.suffix or stem.parent != pathlib.Path("."):
        raise ValueError(f"maskfilestem {self.maskfilestem} shouldn't have '.' or '/' in it")
      filename = stem.with_suffix("."+filetype.lstrip("."))
      if folder is None: folder = self.maskfolder
    else:
      if folder is None: folder = pathlib.Path(".")

    return folder/filename

  def readmask(self, **filekwargs):
    filename = self.maskfilename(**filekwargs)

    filetype = filename.suffix
    if filetype == ".npz":
      dct = np.load(filename)
      return dct["mask"]
    else:
      raise ValueError("Don't know how to deal with mask file type {filetype}")

  @contextlib.contextmanager
  def using_mask(self):
    if self.__using_mask_count == 0:
      self.__mask = self.readmask()
    self.__using_mask_count += 1
    try:
      yield self.__mask
    finally:
      self.__using_mask_count -= 1
      if self.__using_mask_count == 0:
        del self.__mask

class TissueMaskSample(MaskSample):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__using_tissuemask_count = 0

  @classmethod
  @abc.abstractmethod
  def tissuemask(cls, mask): pass

  @contextlib.contextmanager
  def using_tissuemask(self):
    if self.__using_tissuemask_count == 0:
      self.__tissuemask = self.tissuemask(self.readmask())
    self.__using_tissuemask_count += 1
    try:
      yield self.__tissuemask
    finally:
      self.__using_tissuemask_count -= 1
      if self.__using_tissuemask_count == 0:
        del self.__tissuemask

class WriteMaskSampleBase(MaskSample):
  def writemask(self, mask, **filekwargs):
    filename = self.maskfilename(**filekwargs)

    filetype = filename.suffix
    if filetype == ".npz":
      np.savez_compressed(filename, mask=mask)
    else:
      raise ValueError("Don't know how to deal with mask file type {filetype}")

class StitchMaskSampleBase(WriteMaskSampleBase):
  def writemask(self, **filekwargs):
    self.maskfilename(**filekwargs).parent.mkdir(exist_ok=True, parents=True)
    mask = self.stitchmask()
    self.logger.info("saving mask")
    super().writemask(mask, **filekwargs)

  @abc.abstractmethod
  def stitchmask(self): pass

class InformMaskSample(TissueMaskSample):
  @property
  def maskfilestem(self): return "informmask"
  @classmethod
  def tissuemask(cls, mask):
    return mask != 2

class StitchInformMask(ZoomSampleBase, ReadRectanglesDbloadComponentTiff, StitchMaskSampleBase, InformMaskSample):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, layer=9, with_seg=True, **kwargs)

  @property
  def logmodule(self): return "stitchinformmask"

  multilayer = False
  @property
  def rectanglecsv(self): return "fields"
  rectangletype = FieldReadComponentTiff
  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "with_seg": True,
    }

  def stitchmask(self):
    self.logger.info("getting tissue mask")
    mask = np.full(fill_value=2, shape=tuple((self.ntiles * self.zoomtilesize)[::-1]), dtype=np.uint8)
    onepixel = self.onepixel
    nfields = len(self.rectangles)
    for n, field in enumerate(self.rectangles, start=1):
      self.logger.debug(f"putting in mask from field {field.n} ({n}/{nfields})")
      with field.using_image() as im:
        #these lines are copied from zoom.py
        globalx1 = field.mx1 // onepixel * onepixel
        globalx2 = field.mx2 // onepixel * onepixel
        globaly1 = field.my1 // onepixel * onepixel
        globaly2 = field.my2 // onepixel * onepixel
        localx1 = field.mx1 - field.px
        localx2 = localx1 + globalx2 - globalx1
        localy1 = field.my1 - field.py
        localy2 = localy1 + globaly2 - globaly1
        #this part is different
        localx1 = np.round(localx1 / onepixel) * onepixel
        localx2 = np.round(localx2 / onepixel) * onepixel
        localy1 = np.round(localy1 / onepixel) * onepixel
        localy2 = np.round(localy2 / onepixel) * onepixel
        mask[
          floattoint(globaly1/onepixel):floattoint(globaly2/onepixel),
          floattoint(globalx1/onepixel):floattoint(globalx2/onepixel),
        ] = im[
          floattoint(localy1/onepixel):floattoint(localy2/onepixel),
          floattoint(localx1/onepixel):floattoint(localx2/onepixel),
        ]
    return mask
