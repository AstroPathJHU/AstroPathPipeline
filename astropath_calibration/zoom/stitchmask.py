import abc, contextlib, numpy as np, pathlib
from ..alignment.alignmentset import AlignmentSet
from ..alignment.field import FieldReadComponentTiff
from ..baseclasses.sample import MaskSampleBase, ReadRectanglesDbloadComponentTiff, WorkflowSample
from ..utilities.misc import floattoint
from ..zoom.zoom import ZoomSampleBase

class MaskSample(MaskSampleBase):
  """
  Base class for any sample that has a mask that can be loaded from a file.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__using_mask_count = 0

  @classmethod
  @abc.abstractmethod
  def maskfilestem(cls):
    """
    The file stem of the mask file (without the folder or the suffix)
    """
  @classmethod
  def maskfilesuffix(cls):
    return ".npz"

  def maskfilename(self):
    """
    Get the mask filename
    """
    filetype = self.maskfilesuffix()
    stem = pathlib.Path(self.maskfilestem())
    if stem.suffix or stem.parent != pathlib.Path("."):
      raise ValueError(f"maskfilestem {self.maskfilestem()} shouldn't have '.' or '/' in it")
    filename = stem.with_suffix("."+filetype.lstrip("."))
    folder = self.maskfolder
    return folder/filename

  def readmask(self, **filekwargs):
    """
    Read the mask for the sample and return it
    """
    filename = self.maskfilename(**filekwargs)

    filetype = filename.suffix
    if filetype == ".npz":
      dct = np.load(filename)
      return dct["mask"]
    else:
      raise ValueError("Don't know how to deal with mask file type {filetype}")

  @contextlib.contextmanager
  def using_mask(self):
    """
    Context manager for using the mask.  When you enter it for the first time
    it will load the mask. If you enter it again it won't have to load it again.
    When all enters have a matching exit, it will remove it from memory.
    """
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
  """
  Base class for a sample that has a mask for tissue,
  which can be obtained from the main mask. (e.g. if the
  main mask has multiple classifications, the tissue mask
  could be mask == 1)
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__using_tissuemask_count = 0

  @classmethod
  @abc.abstractmethod
  def tissuemask(cls, mask):
    """
    Get the tissue mask from the main mask
    """

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

class WriteMaskSampleBase(MaskSample, WorkflowSample):
  """
  Base class for a sample that creates and writes a mask to file
  """
  def writemask(self, **filekwargs):
    filename = self.maskfilename(**filekwargs)
    filename.parent.mkdir(exist_ok=True, parents=True)

    mask = self.createmask()

    self.logger.info("saving mask")
    filetype = filename.suffix
    if filetype == ".npz":
      np.savez_compressed(filename, mask=mask)
    else:
      raise ValueError("Don't know how to deal with mask file type {filetype}")

  @abc.abstractmethod
  def createmask(self): "create the mask"

  @classmethod
  def getoutputfiles(cls, SlideID, *, maskroot, **otherrootkwargs):
    return [
      maskroot/SlideID/"im3"/"meanimage"/(cls.maskfilestem()+"."+cls.maskfilesuffix().lstrip("."))
    ]

class InformMaskSample(TissueMaskSample):
  """
  Any class that inherits from this can load the inform mask,
  which is layer 9 in the _w_seg component tiff.
  0 is tumor, 1 is healthy tissue, 2 is background.
  The tissue mask is a bool mask that has True for 0 and 1, False for 2
  """
  @classmethod
  def maskfilestem(cls): return "informmask"
  @classmethod
  def tissuemask(cls, mask):
    return mask != 2

class StitchInformMask(ZoomSampleBase, ReadRectanglesDbloadComponentTiff, WriteMaskSampleBase, InformMaskSample):
  """
  Stitch the inform mask together from layer 9 of the component tiffs.
  The implementation is the same as zoom, and the mask will match the
  wsi image to within a pixel (fractional pixel shifts are unavoidable
  because the mask is discrete)
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, layer=9, with_seg=True, **kwargs)

  @classmethod
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

  def createmask(self):
    """
    Stitch the mask together from the component tiffs
    """
    self.logger.info("getting tissue mask")
    #allocate memory for the mask array
    #the default value, which will remain for any pixels that aren't
    #in an HPF, is 2, which corresponds to background
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
        #this part is different, because we can't shift a mask to
        #account for the fractional pixel difference between global
        #and local, so we do the best we can which is to round
        localx1 = np.round(localx1 / onepixel) * onepixel
        localx2 = np.round(localx2 / onepixel) * onepixel
        localy1 = np.round(localy1 / onepixel) * onepixel
        localy2 = np.round(localy2 / onepixel) * onepixel
        if globaly1 < 0:
          localy1 -= globaly1
          globaly1 -= globaly1
        if globalx1 < 0:
          localx1 -= globalx1
          globalx1 -= globalx1
        mask[
          floattoint(globaly1/onepixel):floattoint(globaly2/onepixel),
          floattoint(globalx1/onepixel):floattoint(globalx2/onepixel),
        ] = im[
          floattoint(localy1/onepixel):floattoint(localy2/onepixel),
          floattoint(localx1/onepixel):floattoint(localx2/onepixel),
        ]
    return mask

  @property
  def inputfiles(self):
    return [
      *(r.imagefile for r in self.rectangles),
      self.csv("fields"),
    ]

  @classmethod
  def workflowdependencies(cls):
    return [AlignmentSet] + super().workflowdependencies()
