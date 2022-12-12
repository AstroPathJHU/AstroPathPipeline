import abc, methodtools, numpy as np, pathlib
from ...hpfs.flatfield.config import CONST as FF_CONST
from ...hpfs.flatfield.meanimagesample import MeanImageSampleIm3Tissue
from ...shared.argumentparser import DbloadArgumentParser, MaskArgumentParser
from ...shared.image_masking.maskloader import ThingWithMask, ThingWithTissueMask, ThingWithTissueMaskPolygons
from ...shared.imageloader import ImageLoaderBin, ImageLoaderNpz
from ...shared.logging import ThingWithLogger
from ...shared.rectangle import AstroPathTissueMaskRectangle, IHCTissueMaskRectangle, MaskRectangleBase
from ...shared.rectangletransformation import ImageTransformation
from ...shared.sample import MaskSampleBase, MaskWorkflowSampleBase, ReadRectanglesDbloadSegmentedComponentTiff, TissueSampleBase, TMASampleBase
from ...utilities.img_file_io import im3writeraw
from ...utilities.miscmath import floattoint
from ...utilities.config import CONST as UNIV_CONST
from ..align.alignsample import AlignSample
from ..align.field import Field, FieldReadSegmentedComponentTiffSingleLayer
from ..zoom.wsisamplebase import WSISampleBase

class MaskField(Field, MaskRectangleBase): pass
class AstroPathTissueMaskField(MaskField, AstroPathTissueMaskRectangle): pass
class IHCTissueMaskField(MaskField, IHCTissueMaskRectangle): pass

class MaskSample(MaskSampleBase, WSISampleBase, DbloadArgumentParser, MaskArgumentParser, ThingWithMask, ThingWithLogger):
  """
  Base class for any sample that has a mask that can be loaded from a file.
  """
  @classmethod
  @abc.abstractmethod
  def maskfilestem(cls):
    """
    The file stem of the mask file (without the folder or the suffix)
    """

  @property
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

  @methodtools.lru_cache()
  @property
  def maskloader(self):
    if self.maskfilesuffix == ".npz":
      return ImageLoaderNpz(filename=self.maskfilename, key="mask")
    elif self.maskfilesuffix == ".bin":
      return ImageLoaderBin(filename=self.maskfilename, dimensions=tuple((self.ntiles * self.zoomtilesize)[::-1]))
    else:
      raise ValueError(f"Invalid maskfilesuffix: {self.maskfilesuffix}")

class TissueMaskSample(MaskSample, ThingWithTissueMask):
  """
  Base class for a sample that has a mask for tissue,
  which can be obtained from the main mask. (e.g. if the
  main mask has multiple classifications, the tissue mask
  could be mask == 1)
  """

class TissueMaskSampleWithPolygons(TissueMaskSample, ThingWithTissueMaskPolygons):
  """
  Base class for a sample that has a mask for tissue,
  which can be obtained from the main mask. (e.g. if the
  main mask has multiple classifications, the tissue mask
  could be mask == 1)
  """

class WriteMaskSampleBase(MaskSample, MaskWorkflowSampleBase):
  """
  Base class for a sample that creates and writes a mask to file
  """
  def writemask(self):
    filename = self.maskfilename
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

  def run(self, *args, **kwargs): return self.writemask(*args, **kwargs)

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
  @property
  def tissuemasktransformation(self):
    return ImageTransformation(lambda mask: mask < self.nsegmentations)

class AstroPathTissueMaskSample(TissueMaskSample):
  """
  Any class that inherits from this can load the AstroPath
  tissue mask, which is stored in the im3/meanimage/image_masking
  folder
  """
  @classmethod
  def maskfilestem(cls): return "tissue_mask"
  @property
  def tissuemasktransformation(self):
    return ImageTransformation(lambda mask: mask.astype(bool))

class IHCTissueMaskSample(TissueMaskSample):
  """
  Any class that inherits from this can load the IHC tissue mask
  """
  @classmethod
  def maskfilestem(cls): return "IHC_mask"
  @property
  def tissuemasktransformation(self):
    return ImageTransformation(lambda mask: mask.astype(bool))

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
      if localy2 > im.shape[0] * onepixel:
        globaly2 -= (localy2 - im.shape[0] * onepixel)
        localy2 -= (localy2 - im.shape[0] * onepixel)
      if localx2 > im.shape[1] * onepixel:
        globalx2 -= (localx2 - im.shape[1] * onepixel)
        localx2 -= (localx2 - im.shape[1] * onepixel)
      mask[
        floattoint(float(globaly1/onepixel)):floattoint(float(globaly2/onepixel)),
        floattoint(float(globalx1/onepixel)):floattoint(float(globalx2/onepixel)),
      ] = im[
        floattoint(float(localy1/onepixel)):floattoint(float(localy2/onepixel)),
        floattoint(float(localx1/onepixel)):floattoint(float(localx2/onepixel)),
      ]
    return mask

  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return [AlignSample] + super().workflowdependencyclasses(**kwargs)

class StitchInformMaskSample(StitchMaskSample, ReadRectanglesDbloadSegmentedComponentTiff, InformMaskSample, TissueSampleBase):
  """
  Stitch the inform mask together from layer 9 of the component tiffs.
  The implementation is the same as zoom, and the mask will match the
  wsi image to within a pixel (fractional pixel shifts are unavoidable
  because the mask is discrete)
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, layercomponenttiff="setlater", **kwargs)
    self.setlayerscomponenttiff(layercomponenttiff=self.masklayer)

  @classmethod
  def logmodule(self): return "stitchinformmask"

  multilayer = False
  rectangletype = FieldReadSegmentedComponentTiffSingleLayer

  def inputfiles(self, **kwargs):
    result = [self.csv("fields")]
    if result[0].exists():
      result += [
        r.componenttifffile for r in self.rectangles
      ]
    result += super().inputfiles(**kwargs)
    return result

  @property
  def backgroundvalue(self): return self.nsegmentations
  def getHPFmask(self, field):
    with field.using_component_tiff() as im:
      return im

class StitchAstroPathTissueMaskSampleBase(StitchMaskSample, AstroPathTissueMaskSample):
  """
  Stitch the AstroPath mask together from the bin files.
  The implementation is the same as zoom, and the mask will match the
  wsi image to within a pixel (fractional pixel shifts are unavoidable
  because the mask is discrete)
  """
  @classmethod
  def logmodule(self): return "stitchtissuemask"

  rectangletype = AstroPathTissueMaskField
  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "maskfolder": self.maskfolder,
      "width": self.fwidth,
      "height": self.fheight,
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
    with field.using_tissuemask() as mask: return mask

  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return [MeanImageSampleIm3Tissue] + super().workflowdependencyclasses(**kwargs)

  @property
  def workflowkwargs(self):
    return {**super().workflowkwargs, "skip_masking": False}

class StitchAstroPathTissueMaskSample(StitchAstroPathTissueMaskSampleBase, TissueSampleBase):
  pass
class StitchAstroPathTissueMaskSampleTMA(StitchAstroPathTissueMaskSampleBase, TMASampleBase):
  pass

class StitchIHCTissueMaskSample(StitchMaskSample, IHCTissueMaskSample):
  """
  Stitch the IHC tissue mask together from the bin files.
  The implementation is the same as zoom, and the mask will match the
  wsi image to within a pixel (fractional pixel shifts are unavoidable
  because the mask is discrete)
  """
  @classmethod
  def logmodule(self): return "stitchIHCtissuemask"

  rectangletype = IHCTissueMaskField
  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "ihcmaskfolder": self.ihcmaskfolder,
      "width": self.fwidth,
      "height": self.fheight,
    }

  def inputfiles(self, **kwargs):
    result = [self.csv("fields")]
    if result[0].exists():
      result += [
        r.ihctissuemaskfile for r in self.rectangles
      ]
    return result

  @property
  def backgroundvalue(self): return False
  def getHPFmask(self, field):
    with field.using_tissuemask() as mask: return mask

def astropathtissuemain(args=None):
  StitchAstroPathTissueMaskSample.runfromargumentparser(args=args)
def astropathtissueTMAmain(args=None):
  StitchAstroPathTissueMaskSampleTMA.runfromargumentparser(args=args)

def informmain(args=None):
  StitchInformMaskSample.runfromargumentparser(args=args)

def ihcmain(args=None):
  StitchIHCTissueMaskSample.runfromargumentparser(args=args)
