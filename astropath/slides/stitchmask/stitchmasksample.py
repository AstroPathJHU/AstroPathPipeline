import abc, contextlib, numpy as np, pathlib
from ...hpfs.flatfield.config import CONST as FF_CONST
from ...shared.rectangle import MaskRectangle
from ...shared.argumentparser import DbloadArgumentParser, MaskArgumentParser
from ...shared.sample import MaskSampleBase, ReadRectanglesDbloadComponentTiff, MaskWorkflowSampleBase
from ...shared.image_masking.image_mask import ImageMask
from ...utilities.img_file_io import im3writeraw
from ...utilities.miscmath import floattoint
from ...utilities.config import CONST as UNIV_CONST
from ..align.alignsample import AlignSample
from ..align.field import Field, FieldReadComponentTiff
from ..zoom.zoomsamplebase import ZoomSampleBase

class MaskField(Field, MaskRectangle): pass

class MaskSample(MaskSampleBase, ZoomSampleBase, DbloadArgumentParser, MaskArgumentParser):
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

  def maskfilename(self):
    """
    Get the mask filename
    """
    stem = pathlib.Path(f"{self.SlideID}_{self.maskfilestem()}")
    if stem.suffix or stem.parent != pathlib.Path("."):
      raise ValueError(f"maskfilestem {self.maskfilestem()} shouldn't have '.' or '/' in it")
    filename = stem.with_suffix(self.maskfilesuffix)
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
    elif filetype == ".bin":
      return ImageMask.unpack_tissue_mask(
        filename, tuple((self.ntiles * self.zoomtilesize)[::-1])
      )
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

  @methodtools.lru_cache()
  @property
  def maskpolygons(self):
    with self.using_mask() as mask:
      return findcontoursaspolygons(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, apscale=self.apscale, forgdal=True)

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

  @abc.abstractmethod
  def tissuemask(self, mask):
    """
    Get the tissue mask from the main mask
    """

  @contextlib.contextmanager
  def using_tissuemask(self):
    with contextlib.ExitStack() as stack:
      if self.__using_tissuemask_count == 0:
        self.__tissuemask = self.tissuemask(stack.enter_context(self.using_mask()))
      self.__using_tissuemask_count += 1
      try:
        yield self.__tissuemask
      finally:
        self.__using_tissuemask_count -= 1
        if self.__using_tissuemask_count == 0:
          del self.__tissuemask

class WriteMaskSampleBase(MaskSample, MaskWorkflowSampleBase):
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
    elif filetype == ".bin":
      im3writeraw(filename, np.packbits(mask))
    else:
      raise ValueError("Don't know how to deal with mask file type {filetype}")

  run = writemask

  @abc.abstractmethod
  def createmask(self): "create the mask"

  @classmethod
  def getoutputfiles(cls, SlideID, *, maskroot, maskfilesuffix=None, **otherrootkwargs):
    if maskfilesuffix is None: maskfilesuffix = cls.defaultmaskfilesuffix
    return [
      maskroot/SlideID/UNIV_CONST.IM3_DIR_NAME/UNIV_CONST.MEANIMAGE_DIRNAME/FF_CONST.IMAGE_MASKING_SUBDIR_NAME/pathlib.Path(f"{SlideID}_{cls.maskfilestem()}").with_suffix(maskfilesuffix)
    ]

class InformMaskSample(TissueMaskSample):
  """
  Any class that inherits from this can load the inform mask,
  which is layer 9 in the _w_seg component tiff.
  0 is tumor, 1 is healthy tissue, 2 is background.
  The tissue mask is a bool mask that has True for 0 and 1, False for 2
  """
  @classmethod
  def maskfilestem(cls): return "inform_mask"
  def tissuemask(self, mask):
    return mask < self.nsegmentations

class AstroPathTissueMaskSample(TissueMaskSample):
  """
  Any class that inherits from this can load the AstroPath
  tissue mask, which is stored in the im3/meanimage/image_masking
  folder
  """
  @classmethod
  def maskfilestem(cls): return "tissue_mask"
  def tissuemask(self, mask):
    return mask.astype(bool)

class StitchMaskSample(WriteMaskSampleBase):
  """
  Base class for stitching the global mask together from the individual HPF masks
  """
  rectangletype = Field
  @property
  def rectanglecsv(self): return "fields"

  @property
  @abc.abstractmethod
  def backgroundvalue(self):
    """
    The number that should be filled in the mask for background pixels
    """

  @abc.abstractmethod
  def getHPFmask(self, field):
    """
    Should return the mask from the individual HPF
    """

  def createmask(self):
    """
    Stitch the mask together from the component tiffs
    """
    self.logger.info("getting tissue mask")
    #allocate memory for the mask array
    #the default value, which will remain for any pixels that aren't
    #in an HPF, is 2, which corresponds to background
    mask = np.full(fill_value=self.backgroundvalue, shape=tuple((self.ntiles * self.zoomtilesize)[::-1]), dtype=np.uint8)
    onepixel = self.onepixel
    nfields = len(self.rectangles)
    for n, field in enumerate(self.rectangles, start=1):
      self.logger.debug(f"putting in mask from field {field.n} ({n}/{nfields})")
      im = self.getHPFmask(field)

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
        floattoint(float(globaly1/onepixel)):floattoint(float(globaly2/onepixel)),
        floattoint(float(globalx1/onepixel)):floattoint(float(globalx2/onepixel)),
      ] = im[
        floattoint(float(localy1/onepixel)):floattoint(float(localy2/onepixel)),
        floattoint(float(localx1/onepixel)):floattoint(float(localx2/onepixel)),
      ]
    return mask

  @classmethod
  def workflowdependencyclasses(cls):
    return [AlignSample] + super().workflowdependencyclasses()

class StitchInformMaskSample(StitchMaskSample, ReadRectanglesDbloadComponentTiff, InformMaskSample):
  """
  Stitch the inform mask together from layer 9 of the component tiffs.
  The implementation is the same as zoom, and the mask will match the
  wsi image to within a pixel (fractional pixel shifts are unavoidable
  because the mask is discrete)
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, with_seg=True, layer="setlater", **kwargs)
    self.setlayers(layer=self.masklayer)

  @classmethod
  def logmodule(self): return "stitchinformmask"

  multilayer = False
  rectangletype = FieldReadComponentTiff
  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "with_seg": True,
    }

  def inputfiles(self, **kwargs):
    result = [self.csv("fields")]
    if result[0].exists():
      result += [
        r.imagefile for r in self.rectangles
      ]
    result += super().inputfiles(**kwargs)
    return result

  @property
  def backgroundvalue(self): return self.nsegmentations
  def getHPFmask(self, field):
    with field.using_image() as im:
      return im

class StitchAstroPathTissueMaskSample(StitchMaskSample, AstroPathTissueMaskSample):
  """
  Stitch the AstroPath mask together from the bin files.
  The implementation is the same as zoom, and the mask will match the
  wsi image to within a pixel (fractional pixel shifts are unavoidable
  because the mask is discrete)
  """
  @classmethod
  def logmodule(self): return "stitchtissuemask"

  rectangletype = MaskField
  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "maskfolder": self.maskfolder,
    }

  def inputfiles(self, **kwargs):
    result = [self.csv("fields")]
    if result[0].exists():
      result += [
        r.tissuemaskfile for r in self.rectangles
      ]
    return result

  @property
  def backgroundvalue(self): return False
  def getHPFmask(self, field):
    return ImageMask.unpack_tissue_mask(
        field.tissuemaskfile,
        (
          floattoint(float(self.fheight/self.onepixel)),
          floattoint(float(self.fwidth/self.onepixel)),
        )
      )

def astropathtissuemain(args=None):
  StitchAstroPathTissueMaskSample.runfromargumentparser(args=args)

def informmain(args=None):
  StitchInformMaskSample.runfromargumentparser(args=args)
